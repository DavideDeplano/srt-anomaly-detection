"""
AutoencoderFilter — denoising convolutional autoencoder for only-ON detection
on SRT cadences.

Training strategy
-----------------
The model is trained as a **denoising autoencoder** where cat42 (only-ON)
cadences are the "noise" pattern to be removed.

- For non-cat42 samples: target = input (standard identity reconstruction).
- For cat42 samples:     target = input with ON panels (channels 0,2,4)
                                  replaced by the mean of OFF panels
                                  (channels 1,3,5), broadcast across the
                                  three ON channels. The OFF panels are
                                  left untouched.

The model therefore learns to (a) copy the general cadence morphology,
(b) suppress any signal component that appears only in the ON panels.

At inference on a real candidate the network reconstructs the "cleaned"
version. The reconstruction error concentrates on the ON panels precisely
when the candidate is a true only-ON.

Scoring
-------
raw_score = MSE_ON - MSE_OFF, clipped to 0.
ae_score  = clip( (raw - score_min) / (score_max - score_min), 0, 1 )
            with score_min = p1, score_max = p99 of the raw distribution.

- Real only-ON      -> MSE_ON high (target had ON panels replaced by OFF mean),
                       MSE_OFF low (model reconstructs OFF panels well)
                       -> raw high -> score high.
- Persistent RFI    -> similar residual on ON and OFF -> raw ~ 0 -> score low.
- Only-OFF          -> MSE_ON < MSE_OFF -> raw negative -> clipped to 0.
- Noise / bandpass  -> similar residual on ON and OFF -> raw ~ 0 -> score low.
"""
from __future__ import annotations

import gc
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from src.srtad.config import filters as filters_cfg, paths
from src.srtad.core.candidate import Candidate
from src.srtad.ml.filters.i_filter import IFilter
from src.srtad.ml.models.ae_hypermodel import AEHyperModel
from src.srtad.management.visualizer import Visualizer

_ae_cfg:      Dict = filters_cfg.get("autoencoder", {})
_density_cfg: Dict = filters_cfg.get("density", {})

_ONLY_ON_CATEGORY: int  = int(_density_cfg.get("only_on_category", 42))
_MODEL_PATH:       Path = Path(_ae_cfg.get("model_path", "models/autoencoder.keras"))
_META_PATH:        Path = Path(_ae_cfg.get("meta_path",  "models/autoencoder_meta.joblib"))
_TUNER_DIR:        str  = _ae_cfg.get("tuner_dir", "models/ae_tuner")

_MAX_PER_CATEGORY: int = int(_ae_cfg.get("max_per_category", 3000))

_NORM_P_LOW:  float = float(_ae_cfg.get("norm_p_low", 1.0))
_NORM_P_HIGH: float = float(_ae_cfg.get("norm_p_high", 99.0))

_CALIB_P_LOW:  float = float(_ae_cfg.get("calib_p_low", 1.0))
_CALIB_P_HIGH: float = float(_ae_cfg.get("calib_p_high", 99.0))

_ON_INDICES  = [0, 2, 4]
_OFF_INDICES = [1, 3, 5]

_X_CACHE_PATH    = Path(_TUNER_DIR) / "X_cache.npy"
_Y_CACHE_PATH    = Path(_TUNER_DIR) / "Y_cache.npy"
_X_TRAIN_PATH    = Path(_TUNER_DIR) / "X_train.npy"
_Y_TRAIN_PATH    = Path(_TUNER_DIR) / "Y_train.npy"
_X_VAL_PATH      = Path(_TUNER_DIR) / "X_val.npy"
_Y_VAL_PATH      = Path(_TUNER_DIR) / "Y_val.npy"
_STATS_PATH      = Path(_TUNER_DIR) / "panel_stats.joblib"
_BEST_HPS_PATH   = Path(_TUNER_DIR) / "best_hps.joblib"
_BEST_SCHED_PATH = Path(_TUNER_DIR) / "best_sched.joblib"

_VAL_SPLIT = 0.1


def _compute_global_panel_stats(
    X: np.ndarray, p_low: float = _NORM_P_LOW, p_high: float = _NORM_P_HIGH,
) -> Tuple[np.ndarray, np.ndarray]:
    p_min = np.percentile(X, p_low,  axis=(0, 1, 2))
    p_max = np.percentile(X, p_high, axis=(0, 1, 2))
    return p_min.astype(np.float32), p_max.astype(np.float32)


def _normalize_with_stats(
    X: np.ndarray, p_min: np.ndarray, p_max: np.ndarray,
) -> np.ndarray:
    denom = np.where(p_max - p_min > 0, p_max - p_min, 1.0)
    X = (X - p_min) / denom
    return np.clip(X, 0.0, 1.0).astype(np.float32)


def _normalize_inplace(X: np.ndarray, p_min: np.ndarray, p_max: np.ndarray) -> None:
    denom = np.where(p_max - p_min > 0, p_max - p_min, 1.0)
    X -= p_min
    X /= denom
    np.clip(X, 0.0, 1.0, out=X)


def _build_denoised_target(x_sample: np.ndarray) -> np.ndarray:
    """
    Given a single cat42 sample of shape (H, W, 6) already normalized,
    return the denoised target: ON panels replaced by the mean across
    OFF panels, OFF panels unchanged.
    """
    off_mean = x_sample[..., _OFF_INDICES].mean(axis=-1, keepdims=True)
    y = x_sample.copy()
    for c in _ON_INDICES:
        y[..., c] = off_mean[..., 0]
    return y


def _split_and_save(X: np.ndarray, Y: np.ndarray,
                    val_split: float = _VAL_SPLIT) -> Tuple[Path, Path, Path, Path, int]:
    split = int(len(X) * (1.0 - val_split))
    if not _X_TRAIN_PATH.exists():
        np.save(_X_TRAIN_PATH, X[:split])
        np.save(_Y_TRAIN_PATH, Y[:split])
        np.save(_X_VAL_PATH,   X[split:])
        np.save(_Y_VAL_PATH,   Y[split:])
        print(f"[AUTOENCODER] X/Y_train ({split}) and X/Y_val ({len(X)-split}) saved to disk.")
    else:
        print("[AUTOENCODER] X/Y_train/val already on disk — skipping split.")
    return _X_TRAIN_PATH, _Y_TRAIN_PATH, _X_VAL_PATH, _Y_VAL_PATH, split


def _delta_on_off(diff_sq: np.ndarray) -> float:
    """
    Given squared reconstruction error tensor of shape (H, W, 6),
    return MSE_ON - MSE_OFF (may be negative; clipped downstream).
    """
    mse_on  = float(diff_sq[..., _ON_INDICES].mean())
    mse_off = float(diff_sq[..., _OFF_INDICES].mean())
    return mse_on - mse_off


class AutoencoderFilter(IFilter):
    name: str = "autoencoder"

    def __init__(self) -> None:
        super().__init__()
        self._model:     Optional[keras.Model] = None
        self._score_min: Optional[float] = None
        self._score_max: Optional[float] = None
        self._h:         Optional[int]   = None
        self._w:         Optional[int]   = None
        self._n_panels:  Optional[int]   = None
        self._panel_min: Optional[np.ndarray] = None
        self._panel_max: Optional[np.ndarray] = None
        self._logger = logging.getLogger("srtad.autoencoder")

    def fit(
        self,
        simulated_cadences: Iterable[Tuple[str, np.ndarray, Dict[str, Any]]],
    ) -> None:
        if _MODEL_PATH.exists() and _META_PATH.exists():
            self._logger.info("[AUTOENCODER] Model found — loading.")
            self._load_models()
            return

        Path(_TUNER_DIR).mkdir(parents=True, exist_ok=True)

        if _X_CACHE_PATH.exists() and _Y_CACHE_PATH.exists():
            print("\n[AUTOENCODER] Loading cached X, Y (skipping data collection)...")
            X = np.load(_X_CACHE_PATH)
            Y = np.load(_Y_CACHE_PATH)
            N, H, W, n_panels = X.shape
            self._h, self._w, self._n_panels = H, W, n_panels
            tf.random.set_seed(self._random_state)
            print(f"[AUTOENCODER] X shape from cache: ({N}, {H}, {W}, {n_panels})")

            if _STATS_PATH.exists():
                stats = joblib.load(_STATS_PATH)
                self._panel_min = stats["panel_min"]
                self._panel_max = stats["panel_max"]
            else:
                self._panel_min, self._panel_max = _compute_global_panel_stats(X)
                joblib.dump(
                    {"panel_min": self._panel_min, "panel_max": self._panel_max},
                    _STATS_PATH,
                )

            _split_and_save(X, Y, _VAL_SPLIT)
            del X, Y
            gc.collect()

        else:
            per_cat_count: Dict[int, int] = defaultdict(int)
            kept_arr: List[np.ndarray] = []
            kept_pid: List[int]        = []

            for _cid, tensor, meta in simulated_cadences:
                try:
                    pid = int(meta["pattern_id"])
                    if per_cat_count[pid] >= _MAX_PER_CATEGORY:
                        continue
                    arr = np.asarray(tensor, dtype=np.float32)
                    if arr.ndim != 3 or arr.shape[0] != 6:
                        continue
                except Exception:
                    continue
                per_cat_count[pid] += 1
                kept_arr.append(arr)
                kept_pid.append(pid)

            N = len(kept_arr)
            if N == 0:
                raise RuntimeError(
                    "[AUTOENCODER] No valid training cadences found. Run option 1 first."
                )

            n_cat42 = sum(1 for p in kept_pid if p == _ONLY_ON_CATEGORY)
            self._logger.info(
                "[AUTOENCODER] %d training cadences collected "
                "(cap %d/category, %d categories represented, %d cat42).",
                N, _MAX_PER_CATEGORY, len(per_cat_count), n_cat42,
            )

            H, W = kept_arr[0].shape[1], kept_arr[0].shape[2]
            n_panels = 6

            X = np.empty((N, H, W, n_panels), dtype=np.float32)
            for i, arr in enumerate(kept_arr):
                X[i] = np.transpose(arr, (1, 2, 0))
            pid_arr = np.asarray(kept_pid, dtype=np.int32)
            del kept_arr, kept_pid
            gc.collect()

            self._h, self._w, self._n_panels = H, W, n_panels
            tf.random.set_seed(self._random_state)

            self._panel_min, self._panel_max = _compute_global_panel_stats(X)
            _normalize_inplace(X, self._panel_min, self._panel_max)

            joblib.dump(
                {"panel_min": self._panel_min, "panel_max": self._panel_max},
                _STATS_PATH,
            )
            print("[AUTOENCODER] Panel stats (p{:.0f}/p{:.0f}):".format(_NORM_P_LOW, _NORM_P_HIGH))
            print(f"  panel_min: {self._panel_min}")
            print(f"  panel_max: {self._panel_max}")

            print("[AUTOENCODER] Building denoised targets Y...")
            Y = X.copy()
            cat42_mask = (pid_arr == _ONLY_ON_CATEGORY)
            idx_cat42 = np.where(cat42_mask)[0]
            for i in idx_cat42:
                Y[i] = _build_denoised_target(X[i])
            print(f"[AUTOENCODER] Denoised {len(idx_cat42)} cat42 targets "
                  f"(pannelli ON sostituiti con media pannelli OFF).")

            np.save(_X_CACHE_PATH, X)
            np.save(_Y_CACHE_PATH, Y)
            print(f"[AUTOENCODER] X, Y cache saved -> {_X_CACHE_PATH.parent}  shape={X.shape}")

            _split_and_save(X, Y, _VAL_SPLIT)
            del X, Y, pid_arr
            gc.collect()

        H, W, n_panels = self._h, self._w, self._n_panels

        X_train = np.load(_X_TRAIN_PATH)
        Y_train = np.load(_Y_TRAIN_PATH)
        X_val   = np.load(_X_VAL_PATH)
        Y_val   = np.load(_Y_VAL_PATH)

        if _BEST_HPS_PATH.exists():
            print("\n[AUTOENCODER] Loading saved best_hps (skipping Hyperband)...")
            best_hps = joblib.load(_BEST_HPS_PATH)
        else:
            print("\n[AUTOENCODER] Starting Hyperband tuning...")
            tuner = kt.Hyperband(
                AEHyperModel(H, W, n_panels=n_panels, random_state=self._random_state),
                objective="val_loss",
                max_epochs=50,
                factor=3,
                directory=_TUNER_DIR,
                project_name="ae_tuning",
                seed=self._random_state,
                overwrite=False,
            )
            tuner.search(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                batch_size=128,
                callbacks=[keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True,
                )],
                verbose=1,
            )
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            joblib.dump(best_hps, _BEST_HPS_PATH)
            print(f"[AUTOENCODER] best_hps saved -> {_BEST_HPS_PATH}")

        print("\n[AUTOENCODER] Best hyperparameters:")
        for k in ["n_conv_layers", "filters_1", "filters_2", "latent_dim", "learning_rate"]:
            print(f"  {k:15s}: {best_hps.get(k)}")

        if _BEST_SCHED_PATH.exists():
            print("\n[AUTOENCODER] Loading saved best_sched (skipping grid search)...")
            best_sched = joblib.load(_BEST_SCHED_PATH)
        else:
            print("\n[AUTOENCODER] Grid search epochs x batch_size...")
            n_train = X_train.shape[0]
            n_sub   = max(1000, n_train // 5)
            rng     = np.random.default_rng(self._random_state)
            sub_idx = rng.choice(n_train, size=n_sub, replace=False)
            X_sub   = X_train[sub_idx]
            Y_sub   = Y_train[sub_idx]

            best_sched = {"epochs": 30, "batch_size": 128}
            best_vl    = float("inf")
            hypermodel = AEHyperModel(H, W, n_panels=n_panels, random_state=self._random_state)

            for ep in [30, 50, 80]:
                for bs in [64, 128, 256]:
                    m = hypermodel.build(best_hps)
                    h = m.fit(
                        X_sub, Y_sub,
                        epochs=ep,
                        batch_size=bs,
                        validation_split=0.1,
                        verbose=0,
                        callbacks=[keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=5, restore_best_weights=True,
                        )],
                    )
                    vl = min(h.history["val_loss"])
                    if vl < best_vl:
                        best_vl    = vl
                        best_sched = {"epochs": ep, "batch_size": bs}

            del X_sub, Y_sub
            gc.collect()
            joblib.dump(best_sched, _BEST_SCHED_PATH)
            print(f"[AUTOENCODER] best_sched saved -> {_BEST_SCHED_PATH}")

        best_epochs = best_sched["epochs"]
        best_batch  = best_sched["batch_size"]
        print(f"\n[AUTOENCODER] Best schedule: epochs={best_epochs}, batch_size={best_batch}")

        print("\n[AUTOENCODER] Training final denoising model (X -> Y)...")
        hypermodel  = AEHyperModel(H, W, n_panels=n_panels, random_state=self._random_state)
        self._model = hypermodel.build(best_hps)
        self._model.summary(print_fn=print)

        history = self._model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=best_epochs,
            batch_size=best_batch,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True,
            )],
            verbose=1,
        )

        print("\n[AUTOENCODER] Computing delta ON-OFF scores on training set...")
        X_full = np.load(_X_CACHE_PATH)
        delta_scores = []
        for i in range(0, len(X_full), best_batch):
            batch = X_full[i : i + best_batch]
            recon = self._model.predict(batch, verbose=0)
            diff_sq = (batch - recon) ** 2
            mse_on  = diff_sq[..., _ON_INDICES ].mean(axis=(1, 2, 3))
            mse_off = diff_sq[..., _OFF_INDICES].mean(axis=(1, 2, 3))
            delta_scores.append(mse_on - mse_off)
        delta_scores = np.concatenate(delta_scores)
        del X_full, X_train, Y_train, X_val, Y_val
        gc.collect()

        self._score_min = float(np.percentile(delta_scores, _CALIB_P_LOW))
        self._score_max = float(np.percentile(delta_scores, _CALIB_P_HIGH))
        print(
            f"\n[AUTOENCODER] Delta calibration (training set) | "
            f"score_min(p{_CALIB_P_LOW:.0f})={self._score_min:.4f} | "
            f"score_max(p{_CALIB_P_HIGH:.0f})={self._score_max:.4f}"
        )
        print(f"  Distribuzione delta: median={np.median(delta_scores):.4f}, "
              f"mean={np.mean(delta_scores):.4f}, max={np.max(delta_scores):.4f}")

        Visualizer().plot_autoencoder_diagnostics(history, delta_scores, self._score_max, 99)
        self._save_models()
        self._logger.info("[AUTOENCODER] Training complete.")

    def calculate(self, candidate: Candidate) -> float:
        if self._model is None or self._score_min is None:
            try:
                self._load_models()
            except FileNotFoundError:
                raise RuntimeError("[AUTOENCODER] Model not found. Run option 7 first.")

        cadence = getattr(candidate, "cadence", None)
        if cadence is None:
            return 0.0

        try:
            arr = np.asarray(cadence, dtype=np.float32)
            if arr.ndim != 3 or arr.shape[0] != 6:
                return 0.0

            arr = np.transpose(arr, (1, 2, 0))[None, ...]
            arr = _normalize_with_stats(arr, self._panel_min, self._panel_max)

            recon   = self._model.predict(arr, verbose=0)
            diff_sq = (arr - recon) ** 2
            raw     = _delta_on_off(diff_sq[0])
        except Exception as exc:
            self._logger.warning(
                "[AUTOENCODER] Scoring failed for %s: %s",
                getattr(candidate, "id", "<no-id>"),
                repr(exc),
            )
            return 0.0

        if not math.isfinite(raw):
            return 0.0

        if raw <= 0.0:
            return 0.0

        if self._score_max > self._score_min:
            score = (raw - self._score_min) / (self._score_max - self._score_min)
        else:
            score = 0.0

        return float(np.clip(score, 0.0, 1.0))

    def _save_models(self) -> None:
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        _META_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(_MODEL_PATH)
        joblib.dump(
            {
                "score_min": self._score_min,
                "score_max": self._score_max,
                "h":         self._h,
                "w":         self._w,
                "n_panels":  self._n_panels,
                "panel_min": self._panel_min,
                "panel_max": self._panel_max,
                "scoring":   "on_minus_off_denoiser",
            },
            _META_PATH,
        )
        self._logger.info("[AUTOENCODER] Saved -> %s", _MODEL_PATH)

    def _load_models(self) -> None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"[AUTOENCODER] Model not found: {_MODEL_PATH}")

        if not _META_PATH.exists():
            print(f"[AUTOENCODER] Meta not found — rebuilding from {_STATS_PATH}")
            if not _STATS_PATH.exists():
                raise FileNotFoundError(
                    f"[AUTOENCODER] Cannot rebuild meta: {_STATS_PATH} missing too. "
                    "Run option 7 to retrain."
                )
            stats = joblib.load(_STATS_PATH)
            model_tmp = keras.models.load_model(_MODEL_PATH)
            in_shape = model_tmp.input_shape
            meta = {
                "score_min": 0.0,
                "score_max": 1.0,
                "h": int(in_shape[1]),
                "w": int(in_shape[2]),
                "n_panels":  int(in_shape[3]),
                "panel_min": stats["panel_min"],
                "panel_max": stats["panel_max"],
            }
            joblib.dump(meta, _META_PATH)
            print(f"[AUTOENCODER] Meta rebuilt -> {_META_PATH}")
            del model_tmp

        meta = joblib.load(_META_PATH)
        if "score_min" in meta:
            self._score_min = float(meta["score_min"])
            self._score_max = float(meta["score_max"])
        else:
            self._score_min = float(meta.get("mse_min", 0.0))
            self._score_max = float(meta.get("mse_max", 1.0))
        self._h         = int(meta["h"])
        self._w         = int(meta["w"])
        self._n_panels  = int(meta["n_panels"])
        self._panel_min = np.asarray(meta["panel_min"], dtype=np.float32)
        self._panel_max = np.asarray(meta["panel_max"], dtype=np.float32)
        self._model     = keras.models.load_model(_MODEL_PATH)
        self._logger.info(
            "[AUTOENCODER] Loaded <- %s | score [%.4f, %.4f]",
            _MODEL_PATH, self._score_min, self._score_max,
        )

    def calibrate(self, candidates: List[Candidate]) -> None:
        if self._model is None:
            try:
                self._load_models()
            except FileNotFoundError:
                raise RuntimeError("[AUTOENCODER] Model not loaded. Run option 7 first.")

        delta_list = []
        for c in candidates:
            cadence = getattr(c, "cadence", None)
            if cadence is None:
                continue
            try:
                arr = np.asarray(cadence, dtype=np.float32)
                if arr.ndim != 3 or arr.shape[0] != 6:
                    continue
                arr = np.transpose(arr, (1, 2, 0))[None, ...]
                arr = _normalize_with_stats(arr, self._panel_min, self._panel_max)

                recon   = self._model.predict(arr, verbose=0)
                diff_sq = (arr - recon) ** 2
                d       = _delta_on_off(diff_sq[0])
                if math.isfinite(d):
                    delta_list.append(d)
            except Exception:
                continue

        if not delta_list:
            print("[AUTOENCODER] calibrate(): no valid candidates — keeping original threshold.")
            return

        arr = np.array(delta_list)
        self._score_min = float(np.percentile(arr, _CALIB_P_LOW))
        self._score_max = float(np.percentile(arr, _CALIB_P_HIGH))
        print(
            f"[AUTOENCODER] Calibrated on {len(delta_list)} real candidates (delta ON-OFF) | "
            f"score_min(p{_CALIB_P_LOW:.0f})={self._score_min:.4f} | "
            f"score_max(p{_CALIB_P_HIGH:.0f})={self._score_max:.4f}"
        )
        print(f"  Distribuzione: median={np.median(arr):.4f}, mean={np.mean(arr):.4f}, "
              f"min={np.min(arr):.4f}, max={np.max(arr):.4f}, "
              f"positivi={(arr > 0).sum()}/{len(arr)}")
        self._save_models()