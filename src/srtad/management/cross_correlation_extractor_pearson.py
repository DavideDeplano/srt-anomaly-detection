import numpy as np
from typing import Tuple


class CrossCorrelationExtractor:
    """
    Extract 15 pairwise Pearson correlation features from a cadence (6, H, W).

    Method
    ------
    - Each of the 6 panels is flattened and z-scored.
    - A 6x6 Pearson correlation matrix is computed across panels.
    - The 15 unique off-diagonal correlations (upper triangle, k=1) are returned.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        # Small constant used to avoid division by zero / degenerate standard deviation
        self._eps = float(eps)

    @staticmethod
    def _validate_cadence_shape(cadence: np.ndarray) -> Tuple[int, int, int]:
        """
        Validate that cadence has shape (6, H, W) and return (n_panels, H, W).
        """
        if cadence.ndim != 3:
            raise ValueError(f"Cadence must have shape (6, H, W), got {cadence.shape}")
        n_panels, h, w = cadence.shape
        if n_panels != 6:
            raise ValueError(f"Cadence must contain exactly 6 panels, got {n_panels}")
        return n_panels, h, w

    def _zscore_panel(self, x: np.ndarray) -> np.ndarray:
        """
        Flatten and z-score a single panel.

        Notes
        -----
        If the panel statistics are invalid (non-finite mean/std) or nearly constant
        (std < eps), a zero vector is returned to keep the pipeline numerically stable.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        mu = np.mean(x)
        sigma = np.std(x)
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma < self._eps:
            return np.zeros_like(x, dtype=np.float64)
        return (x - mu) / (sigma + self._eps)

    def extract_features(self, cadence: np.ndarray) -> np.ndarray:
        """
        Compute the 15-dimensional correlation feature vector from a cadence.

        Returns
        -------
        np.ndarray
            A vector of length 15 containing the upper-triangular (k=1) entries of
            the 6x6 Pearson correlation matrix across the 6 panels.
        """
        cadence = np.asarray(cadence, dtype=np.float64)
        n_panels, _, _ = self._validate_cadence_shape(cadence)

        # Stack flattened, z-scored panels: shape (6, H*W)
        X = np.stack([self._zscore_panel(cadence[i]) for i in range(n_panels)], axis=0)

        # Pearson correlation matrix across panels: shape (6, 6)
        C = np.corrcoef(X)

        # Extract the upper triangle without the diagonal -> 15 features
        iu = np.triu_indices(n_panels, k=1)
        features_15d = np.asarray(C[iu], dtype=np.float64)

        # Sanity check: 6 choose 2 = 15
        if features_15d.shape[0] != 15:
            raise RuntimeError(f"Expected 15 CC features, got {features_15d.shape[0]}")

        # Ensure finite outputs
        if not np.all(np.isfinite(features_15d)):
            features_15d = np.nan_to_num(features_15d, nan=0.0, posinf=0.0, neginf=0.0)

        return features_15d
