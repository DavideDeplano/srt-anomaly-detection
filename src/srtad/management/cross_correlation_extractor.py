import numpy as np
from typing import Tuple


class CrossCorrelationExtractor:
    """
    Extract 15 cross-correlation features from a cadence of shape (6, H, W).

    Pipeline
    --------
    1. Each of the 6 panels is flattened into a 1D vector.
    2. Compute the 6x6 Pearson correlation matrix.
    3. Return the 15 upper-triangle (non-diagonal) elements as a feature vector.
    """

    def __init__(self) -> None:
        # Stateless extractor: no configuration required.
        ...

    @staticmethod
    def _validate_cadence_shape(cadence: np.ndarray) -> Tuple[int, int, int]:
        if cadence.ndim != 3:
            raise ValueError(
                f"Cadence must have shape (6, H, W), got {cadence.shape}"
            )

        n_panels, h, w = cadence.shape
        if n_panels != 6:
            raise ValueError(
                f"Cadence must contain exactly 6 panels, got {n_panels}"
            )

        return n_panels, h, w

    def extract_features(self, cadence: np.ndarray) -> np.ndarray:
        cadence = np.asarray(cadence, dtype=np.float64)
        n_panels, h, w = self._validate_cadence_shape(cadence)

        panels_flat = cadence.reshape(n_panels, h * w)

        with np.errstate(invalid="ignore"):
            corr_matrix = np.corrcoef(panels_flat)

        if not np.all(np.isfinite(corr_matrix)):
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        iu = np.triu_indices(n_panels, k=1)
        features_15d = corr_matrix[iu]

        if features_15d.shape[0] != 15:
            raise RuntimeError(
                f"Expected 15 CC features, got {features_15d.shape[0]}"
            )

        return features_15d
