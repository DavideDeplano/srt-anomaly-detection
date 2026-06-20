"""
AutoencoderFilter — convolutional autoencoder for anomaly detection on SRT cadences.

Trains on simulated cadences (excluding only_on_category) and scores real
candidates by reconstruction error. High MSE = anomalous.

All hyperparameters (architecture, lr, epochs, batch_size, threshold percentile)
are searched automatically — only file paths are read from config/default.yaml.

GPU memory strategy
-------------------
The full training array X can exceed 32 GB, which exceeds the VRAM of a single
GPU. To avoid OOM errors, X is never passed directly to model.fit() or
tuner.search(). Instead, X_train and X_val are saved to disk as .npy files and
fed to TensorFlow via memory-mapped generators (mmap_mode="r"), so only one
batch at a time is transferred to the GPU.
"""
from __future__ import annotations

import gc
import logging
import math
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

_X_CACHE_PATH    = Path(_TUNER_DIR) / "X_cache.npy"
_X_TRAIN_PATH    = Path(_TUNER_DIR) / "X_train.npy"
_X_VAL_PATH      = Path(_TUNER_DIR) / "X_val.npy"
_BEST_HPS_PATH   = Path(_TUNER_DIR) / "best_hps.joblib"
_BEST_SCHED_PATH = Path(_TUNER_DIR) / "best_sched.joblib"

# Fraction of data used for validation (rest goes to training)
_VAL_SPLIT = 0.1


def _make_mmap_dataset(
    path: Path,
    h_total: int,
    w: int,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from a memory-mapped .npy file.

    Reads one batch at a time from disk via numpy mmap_mode="r", so the full
    array is never loaded into GPU memory at once. This is the only safe way to
    train on arrays that exceed GPU VRAM.

    Parameters
    ----------
    path : Path
        Path to the .npy file (float32, shape (N, H_total, W, 1)).
    h_total : int
        Height of each sample (panels * H).
    w : int
        Width of each sample.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle sample indices at each epoch.
    seed : int
        Random seed for shuffling.

    Returns
    -------
    ds : tf.data.Dataset
        Yields (batch, batch) tuples for autoencoder training.
    """
    data = np.load(path, mmap_mode="r")
    n    = data.shape[0]

    def gen():
        indices = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        for i in range(0, n, batch_size):
            # Copy batch from mmap into a contiguous float32 array for TF
            batch = np.array(data[indices[i : i + batch_size]], dtype=np.float32)
            yield batch, batch

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, h_total, w, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, h_total, w, 1), dtype=tf.float32),
        ),
    ).prefetch(tf.data.AUTOTUNE)


def _split_and_save(
    X: np.ndarray,
    val_split: float = _VAL_SPLIT,
) -> Tuple[Path, Path, int]:
    """
    Split X into train/val arrays and save them to disk as .npy files.

    Skips saving if the files already exist (checkpoint recovery).

    Parameters
    ----------
    X : np.ndarray
        Full normalised training array, shape (N, H_total, W, 1).
    val_split : float
        Fraction of samples reserved for validation.

    Returns
    -------
    x_train_path : Path
    x_val_path   : Path
    n_train      : int   — number of training samples
    """
    split = int(len(X) * (1.0 - val_split))
    if not _X_TRAIN_PATH.exists():
        np.save(_X_TRAIN_PATH, X[:split])
        np.save(_X_VAL_PATH,   X[split:])
        print(f"[AUTOENCODER] X_train ({split}) and X_val ({len(X)-split}) saved to disk.")
    else:
        print("[AUTOENCODER] X_train/X_val already on disk — skipping split.")
    return _X_TRAIN_PATH, _X_VAL_PATH, split


class AutoencoderFilter(IFilter):
    """
    Convolutional autoencoder filter. Implements IFilter.

    fit()       — tune + train on simulated cadences
    calculate() — return normalised MSE score in [0, 1]
    """

    name: str = "autoencoder"

    def __init__(self) -> None:
        super().__init__()
        self._model:    Optional[keras.Model] = None
        self._mse_min:  Optional[float] = None
        self._mse_max:  Optional[float] = None
        self._h_total:  Optional[int]   = None
        self._w:        Optional[int]   = None
        self._logger = logging.getLogger("srtad.autoencoder")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        simulated_cadences: Iterable[Tuple[str, np.ndarray, Dict[str, Any]]],
    ) -> None:
        """
        Tune and train the autoencoder on simulated cadences.

        Excludes only_on_category from training so the model learns to
        reconstruct "normal" signals and assigns high MSE to only-ON anomalies.

        Checkpoint logic (each stage is skipped if its output already exists):
            0) Data loading + normalisation → X_cache.npy
                                            → X_train.npy / X_val.npy
            1) Hyperband tuning             → best_hps.joblib
            2) Grid search epochs×batch     → best_sched.joblib
            3) Final training + save        → autoencoder.keras
                                            → autoencoder_meta.joblib

        GPU memory note
        ---------------
        X is never passed directly to fit() or tuner.search(). All training
        goes through _make_mmap_dataset(), which reads from disk one batch at
        a time to avoid GPU OOM on large arrays (>22 GB).

        Parameters
        ----------
        simulated_cadences : Iterable
            Yields (cid, tensor, meta) tuples from Dataset.load_simulated_cadences().
            Ignored when X_cache.npy already exists (checkpoint recovery).
        """
        # ---- Stage 0: skip everything if final model already exists ----
        if _MODEL_PATH.exists() and _META_PATH.exists():
            self._logger.info("[AUTOENCODER] Model found — loading.")
            self._load_models()
            return

        Path(_TUNER_DIR).mkdir(parents=True, exist_ok=True)

        # ---- Stage 0a: load or build X ----
        if _X_CACHE_PATH.exists():
            print("\n[AUTOENCODER] Loading cached X (skipping data loading)...")
            # Load with mmap to inspect shape without pulling into RAM
            X_mmap   = np.load(_X_CACHE_PATH, mmap_mode="r")
            N        = X_mmap.shape[0]
            H_total  = X_mmap.shape[1]
            W        = X_mmap.shape[2]
            self._h_total = H_total
            self._w       = W
            tf.random.set_seed(self._random_state)
            print(f"[AUTOENCODER] X shape from cache: ({N}, {H_total}, {W}, 1)")

            # Split to disk if not already done (needed for Hyperband)
            _split_and_save(np.load(_X_CACHE_PATH), _VAL_SPLIT)
            # Free the full array — all further access goes through mmap datasets
            del X_mmap
            gc.collect()

        else:
            # Collect and normalise tensors from simulated cadences
            tensors: List[np.ndarray] = []
            for _cid, tensor, meta in simulated_cadences:
                try:
                    if int(meta["pattern_id"]) == _ONLY_ON_CATEGORY:
                        continue
                    arr = np.asarray(tensor, dtype=np.float32)
                    if arr.ndim != 3 or arr.shape[0] != 6:
                        continue
                except Exception:
                    continue
                tensors.append(arr)

            if not tensors:
                raise RuntimeError(
                    "[AUTOENCODER] No valid training cadences found. Run option 1 first."
                )

            self._logger.info("[AUTOENCODER] %d training cadences collected.", len(tensors))

            # Reshape (N, 6, H, W) → (N, 6H, W, 1) and per-sample normalise
            X = np.stack(tensors, axis=0)
            N, panels, H, W = X.shape
            H_total = panels * H
            X = X.reshape(N, H_total, W, 1)

            X_flat = X.reshape(N, -1)
            X_min  = X_flat.min(axis=1).reshape(N, 1, 1, 1)
            X_max  = X_flat.max(axis=1).reshape(N, 1, 1, 1)
            X      = (X - X_min) / np.where(X_max - X_min > 0, X_max - X_min, 1.0)

            self._h_total = H_total
            self._w       = W
            tf.random.set_seed(self._random_state)

            np.save(_X_CACHE_PATH, X)
            print(f"[AUTOENCODER] X cache saved → {_X_CACHE_PATH}  shape={X.shape}")

            # Split to disk and free RAM — never pass X directly to GPU again
            _split_and_save(X, _VAL_SPLIT)
            del X, tensors
            gc.collect()

        # ---- Stage 1: Hyperband tuning ----
        if _BEST_HPS_PATH.exists():
            print("\n[AUTOENCODER] Loading saved best_hps (skipping Hyperband)...")
            best_hps = joblib.load(_BEST_HPS_PATH)
        else:
            print("\n[AUTOENCODER] Starting Hyperband tuning...")
            tuner = kt.Hyperband(
                AEHyperModel(H_total, W, random_state=self._random_state),
                objective="val_loss",
                max_epochs=50,
                factor=3,
                directory=_TUNER_DIR,
                project_name="ae_tuning",
                seed=self._random_state,
                overwrite=False,
            )

            # Feed data via mmap generators — avoids loading >32 GB onto the GPU
            ds_train = _make_mmap_dataset(
                _X_TRAIN_PATH, H_total, W,
                batch_size=128, shuffle=True, seed=self._random_state,
            )
            ds_val = _make_mmap_dataset(
                _X_VAL_PATH, H_total, W,
                batch_size=128, shuffle=False,
            )

            tuner.search(
                ds_train,
                validation_data=ds_val,
                callbacks=[keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True,
                )],
                verbose=1,
            )
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            joblib.dump(best_hps, _BEST_HPS_PATH)
            print(f"[AUTOENCODER] best_hps saved → {_BEST_HPS_PATH}")

        print("\n[AUTOENCODER] Best hyperparameters:")
        for k in ["n_conv_layers", "filters_1", "filters_2", "latent_dim", "learning_rate"]:
            print(f"  {k:15s}: {best_hps.get(k)}")

        # ---- Stage 2: grid search epochs × batch_size ----
        if _BEST_SCHED_PATH.exists():
            print("\n[AUTOENCODER] Loading saved best_sched (skipping grid search)...")
            best_sched = joblib.load(_BEST_SCHED_PATH)
        else:
            print("\n[AUTOENCODER] Grid search epochs × batch_size...")
            # Use a random subset of X_train (20% or at least 1000 samples)
            X_train_mmap = np.load(_X_TRAIN_PATH, mmap_mode="r")
            n_train      = X_train_mmap.shape[0]
            n_sub        = max(1000, n_train // 5)
            rng          = np.random.default_rng(self._random_state)
            sub_idx      = rng.choice(n_train, size=n_sub, replace=False)
            # Load subset into RAM — it is small enough (≤20% of X_train)
            X_sub        = np.array(X_train_mmap[sub_idx], dtype=np.float32)
            del X_train_mmap
            gc.collect()

            best_sched = {"epochs": 30, "batch_size": 128}
            best_vl    = float("inf")
            hypermodel = AEHyperModel(H_total, W, random_state=self._random_state)

            for ep in [30, 50, 80]:
                for bs in [64, 128, 256]:
                    m = hypermodel.build(best_hps)
                    h = m.fit(
                        X_sub, X_sub,
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

            del X_sub
            gc.collect()
            joblib.dump(best_sched, _BEST_SCHED_PATH)
            print(f"[AUTOENCODER] best_sched saved → {_BEST_SCHED_PATH}")

        best_epochs = best_sched["epochs"]
        best_batch  = best_sched["batch_size"]
        print(f"\n[AUTOENCODER] Best schedule: epochs={best_epochs}, batch_size={best_batch}")

        # ---- Stage 3: final training via mmap datasets ----
        print("\n[AUTOENCODER] Training final model...")
        hypermodel  = AEHyperModel(H_total, W, random_state=self._random_state)
        self._model = hypermodel.build(best_hps)
        self._model.summary(print_fn=print)

        ds_train_final = _make_mmap_dataset(
            _X_TRAIN_PATH, H_total, W,
            batch_size=best_batch, shuffle=True, seed=self._random_state,
        )
        ds_val_final = _make_mmap_dataset(
            _X_VAL_PATH, H_total, W,
            batch_size=best_batch, shuffle=False,
        )

        history = self._model.fit(
            ds_train_final,
            validation_data=ds_val_final,
            epochs=best_epochs,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True,
            )],
            verbose=1,
        )

        # ---- MSE threshold: score all training samples via mmap ----
        # Predict in batches to stay within VRAM budget
        print("\n[AUTOENCODER] Computing MSE scores on training set...")
        X_full_mmap = np.load(_X_CACHE_PATH, mmap_mode="r")
        mse_scores  = []
        for i in range(0, len(X_full_mmap), best_batch):
            batch = np.array(X_full_mmap[i : i + best_batch], dtype=np.float32)
            recon = self._model.predict(batch, verbose=0)
            mse_scores.append(np.mean((batch - recon) ** 2, axis=(1, 2, 3)))
        mse_scores = np.concatenate(mse_scores)
        del X_full_mmap
        gc.collect()

        self._mse_min = float(np.min(mse_scores))

        # Select the percentile threshold that maximises score spread
        best_pct    = 95
        best_spread = -1.0
        for pct in [90, 95, 99]:
            mx = float(np.percentile(mse_scores, pct))
            if mx <= self._mse_min:
                continue
            spread = float(np.std(
                np.clip((mse_scores - self._mse_min) / (mx - self._mse_min), 0, 1)
            ))
            if spread > best_spread:
                best_spread, best_pct = spread, pct

        self._mse_max = float(np.percentile(mse_scores, best_pct))
        print(
            f"\n[AUTOENCODER] Threshold: p{best_pct} | "
            f"mse_min={self._mse_min:.6f} | mse_max={self._mse_max:.6f}"
        )

        Visualizer().plot_autoencoder_diagnostics(history, mse_scores, self._mse_max, best_pct)
        self._save_models()
        self._logger.info("[AUTOENCODER] Training complete.")

    def calculate(self, candidate: Candidate) -> float:
        """
        Return normalised reconstruction MSE for a candidate in [0, 1].

        Higher score = more anomalous (harder to reconstruct).
        Returns 0.0 on any error (missing cadence, model not loaded, etc.).

        Parameters
        ----------
        candidate : Candidate
            Real SRT candidate with a cadence attribute.

        Returns
        -------
        score : float in [0, 1]
        """
        if self._model is None or self._mse_min is None:
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

            panels, H, W = arr.shape
            arr   = arr.reshape(1, panels * H, W, 1)
            a_min = float(arr.min())
            a_max = float(arr.max())
            arr   = (arr - a_min) / ((a_max - a_min) if (a_max - a_min) > 0 else 1.0)

            mse = float(np.mean((arr - self._model.predict(arr, verbose=0)) ** 2))
        except Exception as exc:
            self._logger.warning(
                "[AUTOENCODER] Scoring failed for %s: %s",
                getattr(candidate, "id", "<no-id>"),
                repr(exc),
            )
            return 0.0

        if not math.isfinite(mse):
            return 0.0

        if self._mse_max > self._mse_min:
            score = (mse - self._mse_min) / (self._mse_max - self._mse_min)
        else:
            score = 0.0

        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_models(self) -> None:
        """Save the trained Keras model and MSE metadata to disk."""
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        _META_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(_MODEL_PATH)
        joblib.dump(
            {
                "mse_min": self._mse_min,
                "mse_max": self._mse_max,
                "h_total": self._h_total,
                "w":       self._w,
            },
            _META_PATH,
        )
        self._logger.info("[AUTOENCODER] Saved → %s", _MODEL_PATH)

    def _load_models(self) -> None:
        """Load the trained Keras model and MSE metadata from disk."""
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"[AUTOENCODER] Model not found: {_MODEL_PATH}")
        if not _META_PATH.exists():
            raise FileNotFoundError(f"[AUTOENCODER] Meta not found: {_META_PATH}")

        meta           = joblib.load(_META_PATH)
        self._mse_min  = float(meta["mse_min"])
        self._mse_max  = float(meta["mse_max"])
        self._h_total  = int(meta["h_total"])
        self._w        = int(meta["w"])
        self._model    = keras.models.load_model(_MODEL_PATH)
        self._logger.info(
            "[AUTOENCODER] Loaded ← %s | mse [%.6f, %.6f]",
            _MODEL_PATH, self._mse_min, self._mse_max,
        )

    def calibrate(self, candidates: List[Candidate]) -> None:
        """
        Recalibrate mse_min and mse_max on real candidates.

        The model is trained on simulated float data but real candidates come
        from PNG uint8 images — the MSE distribution is different. Calibrating
        on real data ensures the [0,1] score range is meaningful.

        Parameters
        ----------
        candidates : list of real Candidate objects (with cadence attribute)
        """
        if self._model is None:
          try:
              self._load_models()
          except FileNotFoundError:
              raise RuntimeError("[AUTOENCODER] Model not loaded. Run option 7 first.")

        mse_list = []
        for c in candidates:
            cadence = getattr(c, "cadence", None)
            if cadence is None:
                continue
            try:
                arr = np.asarray(cadence, dtype=np.float32)
                if arr.ndim != 3 or arr.shape[0] != 6:
                    continue
                panels, H, W = arr.shape
                arr   = arr.reshape(1, panels * H, W, 1)
                a_min = float(arr.min())
                a_max = float(arr.max())
                arr   = (arr - a_min) / ((a_max - a_min) if (a_max - a_min) > 0 else 1.0)
                recon = self._model.predict(arr, verbose=0)
                mse   = float(np.mean((arr - recon) ** 2))
                if math.isfinite(mse):
                    mse_list.append(mse)
            except Exception:
                continue

        if not mse_list:
            print("[AUTOENCODER] calibrate(): no valid candidates — keeping original threshold.")
            return

        mse_arr = np.array(mse_list)
        self._mse_min = float(np.percentile(mse_arr, 5))
        self._mse_max = float(np.percentile(mse_arr, 95))
        print(
            f"[AUTOENCODER] Calibrated on {len(mse_list)} real candidates | "
            f"mse_min(p5)={self._mse_min:.6f} | mse_max(p95)={self._mse_max:.6f}"
        )
        self._save_models()