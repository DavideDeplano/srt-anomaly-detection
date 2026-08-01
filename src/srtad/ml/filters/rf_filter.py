"""
RandomForestFilter — supervised classifier on handcrafted features.

Trained on synthetic cadences with binary labels:
    positive = cat42 (only-ON)
    negative = any other pattern (63 categories, balanced)

Critical: synthetic cadences are preprocessed with the same pipeline used
for real candidates (Dataset._preprocess_spectrogram: time normalization,
DC removal, B-spline bandpass correction). Without this alignment the RF
sees synthetic values in raw scale (~1e6) while real candidates arrive
normalized around 1.0, producing a catastrophic domain shift.
"""
from __future__ import annotations

import gc
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.srtad.config import filters as filters_cfg, paths
from src.srtad.core.candidate import Candidate
from src.srtad.core.dataset import _preprocess_spectrogram
from src.srtad.ml.filters.i_filter import IFilter
from src.srtad.ml.models.rf_features import (
    extract_features,
    extract_features_batch,
    FEATURE_NAMES,
    N_FEATURES,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

_rf_cfg:      Dict = filters_cfg.get("rf_filter", {})
_density_cfg: Dict = filters_cfg.get("density", {})

_ONLY_ON_CATEGORY: int  = int(_density_cfg.get("only_on_category", 42))
_MODEL_PATH:       Path = Path(_rf_cfg.get("model_path", "models/rf_filter/classifier.joblib"))
_META_PATH:        Path = Path(_rf_cfg.get("meta_path",  "models/rf_filter/meta.joblib"))

_N_CAT42:     int = int(_rf_cfg.get("n_cat42", 20000))
_N_PER_OTHER: int = int(_rf_cfg.get("n_per_other", 320))

_N_ESTIMATORS_GRID: List[int] = list(_rf_cfg.get("n_estimators_grid", [200, 500]))
_MAX_DEPTH_GRID:    List[int] = list(_rf_cfg.get("max_depth_grid",    [6, 8, 12]))
_CV_FOLDS:          int       = int(_rf_cfg.get("cv_folds", 5))

_NORM_P_LOW:  float = float(_rf_cfg.get("norm_p_low",  1.0))
_NORM_P_HIGH: float = float(_rf_cfg.get("norm_p_high", 99.0))

_CAD_FILENAME_RX = re.compile(r"^cadence_(\d+)_pattern(\d+)\.npy$")


def _load_and_preprocess(path: Path) -> np.ndarray:
    """
    Load a synthetic cadence .npy and apply the SAME preprocessing used for
    real candidates (Dataset._preprocess_spectrogram on each of the 6 panels).

    Returns a (6, H, W) tensor with values on the same scale as real
    candidates in Candidate.cadence.
    """
    raw = np.load(path)
    out = np.empty_like(raw, dtype=np.float32)
    for i in range(6):
        out[i] = _preprocess_spectrogram(raw[i]).astype(np.float32)
    return out


class RandomForestFilter(IFilter):
    name: str = "rf_filter"

    def __init__(self) -> None:
        super().__init__()
        self._model:     Optional[RandomForestClassifier] = None
        self._panel_min: Optional[np.ndarray] = None
        self._panel_max: Optional[np.ndarray] = None
        self._best_params: Optional[Dict[str, int]] = None
        self._cv_auc:      Optional[float] = None
        self._logger = logging.getLogger("srtad.rf_filter")

    def fit(
        self,
        simulated_cadences: Iterable[Tuple[str, np.ndarray, Dict[str, Any]]],
    ) -> None:
        if _MODEL_PATH.exists() and _META_PATH.exists():
            self._logger.info("[RF_FILTER] Model found — loading.")
            self._load_models()
            return

        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        _META_PATH.parent.mkdir(parents=True, exist_ok=True)

        cadences_dir = Path(paths["data"]) / filters_cfg.get("simulation", {}).get(
            "output_cadences_dir", "simulated/cadences"
        )
        if not cadences_dir.exists():
            fallback = Path("data/simulated/cadences")
            if fallback.exists():
                cadences_dir = fallback
            else:
                raise FileNotFoundError(
                    f"[RF_FILTER] Cadences directory not found. "
                    f"Tried: {cadences_dir}, {fallback}. Run option 1 first."
                )

        self._logger.info(
            "[RF_FILTER] Scanning %s (target: %d cat42 + %d*63 other)...",
            cadences_dir, _N_CAT42, _N_PER_OTHER,
        )

        cat42_files: List[Path] = []
        other_files: Dict[int, List[Path]] = defaultdict(list)
        scanned = 0

        for f in cadences_dir.iterdir():
            scanned += 1
            m = _CAD_FILENAME_RX.match(f.name)
            if not m:
                continue
            pid = int(m.group(2))
            if pid == _ONLY_ON_CATEGORY:
                if len(cat42_files) < _N_CAT42:
                    cat42_files.append(f)
            else:
                if len(other_files[pid]) < _N_PER_OTHER:
                    other_files[pid].append(f)
            total_other = sum(len(v) for v in other_files.values())
            if len(cat42_files) >= _N_CAT42 and total_other >= _N_PER_OTHER * 63:
                break
            if scanned % 50000 == 0:
                print(f"[RF_FILTER] scanned={scanned} | cat42={len(cat42_files)} | other={total_other}")

        total_other = sum(len(v) for v in other_files.values())
        print(f"[RF_FILTER] Selected: cat42={len(cat42_files)}, other={total_other}, scanned={scanned}")

        all_paths = cat42_files + [f for flist in other_files.values() for f in flist]
        n_total = len(all_paths)
        print(f"[RF_FILTER] Loading and preprocessing {n_total} .npy files (parallel)...")

        loaded = Parallel(n_jobs=-1, prefer="processes", verbose=1)(
            delayed(_load_and_preprocess)(p) for p in all_paths
        )

        X = np.stack(
            [np.transpose(a, (1, 2, 0)) for a in loaded],
            axis=0,
        ).astype(np.float32)
        labels = np.array(
            [1] * len(cat42_files) + [0] * total_other,
            dtype=np.int32,
        )
        del loaded, cat42_files, other_files
        gc.collect()

        print(f"[RF_FILTER] X shape: {X.shape}, cat42={int(labels.sum())}, other={int((labels==0).sum())}")
        print(f"[RF_FILTER] X value range (post-preprocessing): "
              f"min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}")

        self._panel_min = np.percentile(X, _NORM_P_LOW,  axis=(0, 1, 2)).astype(np.float32)
        self._panel_max = np.percentile(X, _NORM_P_HIGH, axis=(0, 1, 2)).astype(np.float32)
        denom = np.where(self._panel_max - self._panel_min > 0,
                         self._panel_max - self._panel_min, 1.0)
        X = np.clip((X - self._panel_min) / denom, 0.0, 1.0).astype(np.float32)
        print(f"[RF_FILTER] panel_min: {self._panel_min}")
        print(f"[RF_FILTER] panel_max: {self._panel_max}")

        print(f"[RF_FILTER] Extracting {N_FEATURES}D features...")
        F = extract_features_batch(X)
        del X
        gc.collect()
        print(f"[RF_FILTER] Feature matrix: {F.shape}")

        print(f"[RF_FILTER] Grid search "
              f"n_estimators={_N_ESTIMATORS_GRID} x max_depth={_MAX_DEPTH_GRID} "
              f"({_CV_FOLDS}-fold CV)...")
        skf = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=self._random_state)
        best_score  = -1.0
        best_params = {"n_estimators": _N_ESTIMATORS_GRID[0], "max_depth": _MAX_DEPTH_GRID[0]}
        for n_est in _N_ESTIMATORS_GRID:
            for depth in _MAX_DEPTH_GRID:
                clf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    random_state=self._random_state,
                    n_jobs=-1,
                )
                scores = cross_val_score(clf, F, labels, cv=skf, scoring="roc_auc", n_jobs=-1)
                mean = float(scores.mean())
                std  = float(scores.std())
                print(f"[RF_FILTER]   n_est={n_est} max_depth={depth} | "
                      f"CV AUC = {mean:.4f} +/- {std:.4f}")
                if mean > best_score:
                    best_score  = mean
                    best_params = {"n_estimators": n_est, "max_depth": depth}

        print(f"[RF_FILTER] Best params: {best_params} | CV AUC = {best_score:.4f}")

        self._model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            random_state=self._random_state,
            n_jobs=-1,
        )
        self._model.fit(F, labels)
        self._best_params = best_params
        self._cv_auc      = best_score

        print(f"[RF_FILTER] Top 8 feature importance:")
        order = np.argsort(self._model.feature_importances_)[::-1]
        for i in order[:8]:
            print(f"[RF_FILTER]   {FEATURE_NAMES[i]:35s} {self._model.feature_importances_[i]:.4f}")

        self._save_models()
        self._logger.info("[RF_FILTER] Training complete.")

    def calculate(self, candidate: Candidate) -> float:
        if self._model is None:
            try:
                self._load_models()
            except FileNotFoundError:
                raise RuntimeError("[RF_FILTER] Model not found. Train filter first.")

        cadence = getattr(candidate, "cadence", None)
        if cadence is None:
            return 0.0

        try:
            arr = np.asarray(cadence, dtype=np.float32)
            if arr.ndim != 3 or arr.shape[0] != 6:
                return 0.0

            arr = np.transpose(arr, (1, 2, 0))
            denom = np.where(self._panel_max - self._panel_min > 0,
                             self._panel_max - self._panel_min, 1.0)
            arr = np.clip((arr - self._panel_min) / denom, 0.0, 1.0)

            feats = extract_features(arr).reshape(1, -1)
            proba = self._model.predict_proba(feats)[0, 1]
        except Exception as exc:
            self._logger.warning(
                "[RF_FILTER] Scoring failed for %s: %s",
                getattr(candidate, "id", "<no-id>"),
                repr(exc),
            )
            return 0.0

        return float(np.clip(proba, 0.0, 1.0))

    def _save_models(self) -> None:
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        _META_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, _MODEL_PATH)
        joblib.dump(
            {
                "panel_min":         self._panel_min,
                "panel_max":         self._panel_max,
                "feature_names":     FEATURE_NAMES,
                "best_params":       self._best_params,
                "cv_auc":            self._cv_auc,
                "feature_importance": (
                    self._model.feature_importances_.tolist()
                    if self._model is not None else None
                ),
                "n_cat42_training":  _N_CAT42,
                "n_per_other":       _N_PER_OTHER,
                "preprocessing":     "dataset_preprocess_spectrogram",
            },
            _META_PATH,
        )
        self._logger.info("[RF_FILTER] Saved -> %s", _MODEL_PATH)

    def _load_models(self) -> None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"[RF_FILTER] Model not found: {_MODEL_PATH}")
        if not _META_PATH.exists():
            raise FileNotFoundError(f"[RF_FILTER] Meta not found: {_META_PATH}")

        self._model     = joblib.load(_MODEL_PATH)
        meta            = joblib.load(_META_PATH)
        self._panel_min = np.asarray(meta["panel_min"], dtype=np.float32)
        self._panel_max = np.asarray(meta["panel_max"], dtype=np.float32)
        self._best_params = meta.get("best_params")
        self._cv_auc      = meta.get("cv_auc")

        self._logger.info(
            "[RF_FILTER] Loaded <- %s | best_params=%s | CV AUC=%.4f",
            _MODEL_PATH,
            self._best_params,
            (self._cv_auc if self._cv_auc is not None else float('nan')),
        )