from typing import Iterable, List, Dict
from pathlib import Path
import logging
import math

import joblib
import numpy as np
from umap import UMAP

from .i_filter import IFilter
from ...core.candidate import Candidate
from ...config import filters

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

class SimilarityFilter(IFilter):
    """
    ON/OFF similarity filter based on panel embeddings in a UMAP space

    Concept
    -------
    - Each candidate cadence contains 6 panels (ON/OFF pattern)
    - Each panel is downsampled from (16, 80) to (8, 16) using block averaging
    - Each downsampled panel is flattened and embedded with UMAP
    - Similarity is computed from distances in the embedding space using a fixed pattern:
        ON indices  = [0, 2, 4]
        OFF indices = [1, 3, 5]

    Raw score
    ---------
    The raw score is a ratio of mean distances:
        ratio = mean_d(ON-OFF) / (mean_d(ON-ON) + eps)

    Interpretation
    --------------
    - ON panels should be close to each other (small ON-ON distance)
    - ON panels should be far from OFF panels (large ON-OFF distance)
    - Therefore, larger ratios indicate stronger ON/OFF separation and are scored higher

    Normalization
    -------------
    A global min/max range of raw ratios is computed on training candidates and used to
    scale scores into [0, 1] during inference
    """
    name: str = "similarity"

    def __init__(self) -> None:
        """
        Initialize SimilarityFilter from YAML configuration

        Configuration keys (filters.similarity)
        --------------------------------------
        - umap_model_path: path to persist the fitted UMAP model
        - n_neighbors, min_dist, umap_metric, n_components: UMAP hyperparameters
        """
        super().__init__()

        similarity_cfg = filters["similarity"]

        self._logger = logging.getLogger("srtad.similarity")
        self._use_tqdm = True

        # UMAP persistence paths
        self._umap_model_path = Path(similarity_cfg["umap_model_path"])

        # UMAP hyperparameters 
        self._n_neighbors = int(similarity_cfg["n_neighbors"])
        self._min_dist = float(similarity_cfg["min_dist"])
        self._metric = str(similarity_cfg["umap_metric"])
        self._n_components = int(similarity_cfg["n_components"])

        # UMAP model used to embed individual panels
        self._umap = UMAP(
            n_neighbors=self._n_neighbors,
            min_dist=self._min_dist,
            n_components=self._n_components,
            metric=self._metric,
            random_state=self.random_state,
        )

        # Downsampling factors expected for the pipeline
        # Frequency: 80 -> 16 via factor 5
        # Time:      16 ->  8 via factor 2
        self._raw_min: float | None = None
        self._raw_max: float | None = None

        # Downsampling factors (MANDATORY: 80->16 freq, 16->8 time)
        self._freq_downsample: int = 5
        self._time_downsample: int = 2

        # Fixed ON/OFF indices for the 6-panel cadence pattern
        self._on_indices: List[int] = [0, 2, 4]
        self._off_indices: List[int] = [1, 3, 5]

    def fit(self, candidates: Iterable[Candidate]) -> None:
        """
        Fit the UMAP model on downsampled panels and compute scaling statistics

        Steps
        -----
        1) Collect downsampled panels from all training candidates
        2) Flatten panels into feature vectors
        3) Train UMAP on a fixed fraction of collected panels (subsampling)
        4) Compute raw score min/max on the full candidate set for normalization
        5) Save UMAP model and metadata to disk
        """
        # Materialize candidates because the iterable is used multiple times
        candidates_list: List[Candidate] = list(candidates)

        if not candidates_list:
            msg = "[SIMILARITY FILTER] fit: empty candidate list; cannot train."
            self._logger.error(msg)
            raise RuntimeError(msg)

        self._logger.info(
            "[SIMILARITY FILTER] Training mode: fitting UMAP on downsampled panels..."
        )

        # Panel-level training data for UMAP
        # Each cadence contributes up to 6 samples (one per panel)
        X_panels: List[np.ndarray] = []

        iterator = candidates_list
        if self._use_tqdm and tqdm is not None:
            iterator = tqdm(
                candidates_list,
                desc="Collecting panels for UMAP",
                leave=False,
            )

        for c in iterator:
            cadence = getattr(c, "cadence", None)
            if cadence is None:
                continue

            cadence = np.asarray(cadence, dtype=float)
            if cadence.ndim != 3 or cadence.shape[0] != 6:
                continue

            # Downsample cadence panels to the expected (6, 8, 16) representation
            try:
                ds = self._downsample_cadence(cadence)  # (6, 8, 16)
            except Exception as exc:
                self._logger.warning(
                    "[SIMILARITY FILTER] Downsampling failed for candidate %s: %s",
                    getattr(c, "id", "<no-id>"),
                    repr(exc),
                )
                continue

            # Flatten each panel for UMAP input
            panels_flat = ds.reshape(6, -1)  # (6, 128)
            for i in range(panels_flat.shape[0]):
                X_panels.append(panels_flat[i])

        if not X_panels:
            msg = "[SIMILARITY FILTER] fit: no valid panels collected; cannot train UMAP."
            self._logger.error(msg)
            raise RuntimeError(msg)

        X_arr = np.asarray(X_panels, dtype=float)  # (N_panels, features)
        n_panels, n_features = X_arr.shape

        self._logger.info(
            "[SIMILARITY FILTER] Collected %d panels with %d features each.",
            n_panels,
            n_features,
        )

        # Subsample panels for UMAP training for speed
        # The scoring still uses all panels at inference time via transform
        train_fraction = 0.20
        n_train = max(1, int(train_fraction * n_panels))

        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n_panels, size=n_train, replace=False)
        X_train = X_arr[idx]
        
        self._logger.info(
            "[SIMILARITY FILTER] Training UMAP on %d panels (fixed 20%% of %d total).",
            n_train, n_panels
        )

        # Fit UMAP on the sampled panel vectors
        self._logger.info("[SIMILARITY FILTER] Fitting UMAP...")
        self._umap.fit(X_train)

        # Compute raw similarity score scaling range on the full candidate list
        self._logger.info(
            "[SIMILARITY FILTER] Computing raw similarity score range for scaling..."
        )
        self._compute_scaling(candidates_list)

        # Persist UMAP model and metadata (scaling + configuration)
        self._save_models()
        self._logger.info("[SIMILARITY FILTER] Fit complete.")

    def _compute_scaling(self, candidates: Iterable[Candidate]) -> None:
        """
        Compute global min/max of raw similarity ratios across training candidates
        """
        raw_scores: List[float] = []

        for c in candidates:
            raw = self._raw_similarity_score(c)
            if math.isfinite(raw):
                raw_scores.append(raw)

        if not raw_scores:
            self._raw_min = 0.0
            self._raw_max = 0.0
            self._logger.warning(
                "[SIMILARITY FILTER] No valid raw scores for scaling; "
                "using degenerate [0, 0] range."
            )
            return

        self._raw_min = float(np.min(raw_scores))
        self._raw_max = float(np.max(raw_scores))

        self._logger.info(
            "[SIMILARITY FILTER] Raw similarity score range: min=%.4f, max=%.4f",
            self._raw_min,
            self._raw_max,
        )

    def calculate(self, candidate: Candidate) -> float:
        """
        Compute the ON/OFF similarity score for a candidate

        Interpretation
        --------------
        Higher score indicates ON panels are mutually close and separated from OFF panels
        """
        # Auto-load model and scaling if not initialized
        if self._raw_min is None or self._raw_max is None:
            try:
                self._load_models()
            except FileNotFoundError:
                raise RuntimeError(
                    "[SIMILARITY FILTER] Model files not found; call fit(...) first "
                    "to train and save the UMAP model."
                )

        raw = self._raw_similarity_score(candidate)

        if not math.isfinite(raw):
            return 0.0

        # Linear scaling into [0, 1]
        if self._raw_max > self._raw_min:
            score = (raw - self._raw_min) / (self._raw_max - self._raw_min)
        else:
            score = 0.0

        return float(np.clip(score, 0.0, 1.0))
    
    def _raw_similarity_score(self, candidate: Candidate) -> float:
        """
        Compute the raw similarity ratio for a single candidate

        Definition
        ----------
        ratio = mean_d(ON-OFF) / (mean_d(ON-ON) + eps)

        Guards
        ------
        - Returns 0.0 if cadence is missing, malformed, or distances cannot be computed
        """
        cadence = getattr(candidate, "cadence", None)
        if cadence is None:
            return 0.0

        cadence = np.asarray(cadence, dtype=float)
        if cadence.ndim != 3 or cadence.shape[0] != 6:
            return 0.0

        try:
            # Downsample cadence panels and flatten for UMAP transform
            ds = self._downsample_cadence(cadence)          # (6, 16, 8)
            panels_flat = ds.reshape(6, -1)                 # (6, 128)

            # Safety check: expected flattened panel dimensionality
            if panels_flat.shape[1] != 128:
                self._logger.warning(
                    "[SIMILARITY FILTER] Unexpected feature size %d for candidate %s",
                    panels_flat.shape[1],
                    getattr(candidate, "id", "<no-id>")
                )
                return 0.0

            # Embed each panel into UMAP space
            Z = self._umap.transform(panels_flat)           # (6, n_components)

        except Exception as exc:
            self._logger.warning(
                "[SIMILARITY FILTER] Embedding failed for candidate %s: %s",
                getattr(candidate, "id", "<no-id>"),
                repr(exc),
            )
            return 0.0

        # Validate ON/OFF indices against the embedded panel count
        on_idx = [i for i in self._on_indices if 0 <= i < Z.shape[0]]
        off_idx = [i for i in self._off_indices if 0 <= i < Z.shape[0]]

        # Need at least two ON panels for ON-ON distances and at least one OFF panel
        if len(on_idx) < 2 or len(off_idx) == 0:
            return 0.0

        Z_on = Z[on_idx]    
        Z_off = Z[off_idx]  

        # Pairwise distances among ON panels (upper triangle)
        on_on_dists: List[float] = []
        for i in range(len(Z_on)):
            for j in range(i + 1, len(Z_on)):
                d = np.linalg.norm(Z_on[i] - Z_on[j])
                if math.isfinite(d):
                    on_on_dists.append(float(d))

        # Pairwise distances between ON and OFF panels
        on_off_dists: List[float] = []
        for i in range(len(Z_on)):
            for j in range(len(Z_off)):
                d = np.linalg.norm(Z_on[i] - Z_off[j])
                if math.isfinite(d):
                    on_off_dists.append(float(d))

        if not on_on_dists or not on_off_dists:
            return 0.0

        mean_on_on = float(np.mean(on_on_dists))
        mean_on_off = float(np.mean(on_off_dists))

        # Epsilon avoids division by zero when ON panels collapse into nearly identical points
        eps = 1e-6
        ratio = mean_on_off / (mean_on_on + eps)

        if not math.isfinite(ratio):
            return 0.0

        return float(ratio)

    def _downsample_cadence(self, cadence: np.ndarray) -> np.ndarray:
        """
        Downsample a cadence from (6, H, W) to (6, H/2, W/5) using block averaging

        Constraints
        -----------
        This implementation assumes input panels are already standardized to H=16 and W=80
        because it requires H divisible by 2 and W divisible by 5
        """
        if cadence.ndim != 3 or cadence.shape[0] != 6:
            raise ValueError(
                f"Invalid cadence shape {cadence.shape}; expected (6, H, W)."
            )

        _, H, W = cadence.shape

        # Reshape into blocks: (6, new_H, time_factor, new_W, freq_factor)
        time_factor = self._time_downsample
        freq_factor = self._freq_downsample 

        if H % time_factor != 0 or W % freq_factor != 0:
            raise ValueError(
                f"Cadence shape ({H}, {W}) is not divisible by "
                f"({time_factor}, {freq_factor}). Expected H=16, W=80."
            )

        new_H = H // time_factor    # New time dimension (e.g., 16 // 2 = 8)
        new_W = W // freq_factor    # New frequency dimension (e.g., 80 // 5 = 16)
        
        # Reshape into blocks and average
        # The shape decomposition must be: (N_cadences, N_H_blocks, H_factor, N_W_blocks, W_factor)
        cad_reshaped = cadence.reshape(
            6,
            new_H,
            time_factor, # Axis to average for Time reduction (H)
            new_W,
            freq_factor, # Axis to average for Frequency reduction (W)
        )
        
        # Average within each block to reduce resolution
        ds = cad_reshaped.mean(axis=(2, 4)) 

        # Expected output shape: (6, 8, 16) for input (6, 16, 80)
        return ds

    def _save_models(self) -> None:
        """
        Save UMAP model and metadata to disk

        Saved artifacts
        --------------
        - UMAP model: used to embed panels at inference time
        - Metadata: scaling range and configuration required for consistent scoring
        """
        path = self._umap_model_path

        if path is None or not str(path):
            self._logger.warning(
                "[SIMILARITY FILTER] umap_model_path not set; skipping save."
            )
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = path.with_suffix(".meta.joblib")

        state = {
            "raw_min": self._raw_min,
            "raw_max": self._raw_max,
            "freq_downsample": self._freq_downsample,
            "time_downsample": self._time_downsample,
            "on_indices": self._on_indices,
            "off_indices": self._off_indices,
        }

        self._logger.info(
            "[SIMILARITY FILTER] Saving UMAP model to %s and metadata to %s",
            path,
            meta_path,
        )
        joblib.dump(self._umap, path)
        joblib.dump(state, meta_path)

    def _load_models(self) -> None:
        """
        Load UMAP model and metadata from disk

        Restores
        --------
        - UMAP model for panel embeddings
        - raw_min/raw_max for score scaling
        - downsampling factors and ON/OFF indices for consistency
        """
        path = self._umap_model_path

        if path is None or not str(path):
            msg = "[SIMILARITY FILTER] umap_model_path is empty in inference mode."
            self._logger.error(msg)
            raise RuntimeError(msg)

        meta_path = path.with_suffix(".meta.joblib")

        if not path.exists():
            msg = f"[SIMILARITY FILTER] UMAP model file not found: {path}"
            self._logger.error(msg)
            raise FileNotFoundError(msg)

        if not meta_path.exists():
            msg = f"[SIMILARITY FILTER] Metadata file not found: {meta_path}"
            self._logger.error(msg)
            raise FileNotFoundError(msg)

        self._logger.info(
            "[SIMILARITY FILTER] Loading UMAP model from %s and metadata from %s",
            path,
            meta_path,
        )
        self._umap = joblib.load(path)

        if not hasattr(self._umap, "transform"):
            raise RuntimeError("[SIMILARITY FILTER] Loaded UMAP model has no transform().")
        
        state = joblib.load(meta_path)

        self._raw_min = state.get("raw_min", 0.0)
        self._raw_max = state.get("raw_max", 0.0)

        self._freq_downsample = int(
            state.get("freq_downsample", self._freq_downsample)
        )
        self._time_downsample = int(
            state.get("time_downsample", self._time_downsample)
        )
        self._on_indices = list(state.get("on_indices", self._on_indices))
        self._off_indices = list(state.get("off_indices", self._off_indices))
