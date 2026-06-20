import numpy as np
from typing import Tuple

from scipy.signal import fftconvolve


class CrossCorrelationExtractor:
    """
    Extract 15 pairwise cross-correlation features from a cadence (6, H, W).

    Method
    ------
    - Each of the 6 panels is z-scored.
    - For each of the 15 unique panel pairs, the full 2D normalized
      cross-correlation is computed via FFT and the MAXIMUM over a
      restricted lag window is taken as the pair feature.

    Rationale
    ---------
    The maximum of the cross-correlation matrix is invariant to relative
    shifts between panels. A narrowband signal that drifts in frequency
    between observations still produces a high cross-correlation peak at a
    non-zero lag, whereas the zero-lag Pearson coefficient vanishes as soon
    as the signal positions stop overlapping. This follows the feature
    definition in Pardo et al. (2025), where per-pair statistics are
    tabulated from the full CC matrix.

    The lag window is restricted (default: |dt| <= H/4, |df| <= W/2) to
    avoid spurious maxima from small-overlap regions at extreme lags.
    """

    def __init__(self,
                 eps: float = 1e-8,
                 max_time_lag_frac: float = 0.05,
                 max_freq_lag_frac: float = 0.1) -> None:
        # Small constant used to avoid division by zero / degenerate standard deviation
        self._eps = float(eps)
        # Lag window restriction, as fractions of panel height/width
        self._max_time_lag_frac = float(max_time_lag_frac)
        self._max_freq_lag_frac = float(max_freq_lag_frac)

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
        Z-score a single panel (2D), preserving its shape.

        Notes
        -----
        If the panel statistics are invalid (non-finite mean/std) or nearly constant
        (std < eps), a zero array is returned to keep the pipeline numerically stable.
        """
        x = np.asarray(x, dtype=np.float64)
        mu = np.mean(x)
        sigma = np.std(x)
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma < self._eps:
            return np.zeros_like(x, dtype=np.float64)
        return (x - mu) / (sigma + self._eps)

    def _ncc_max(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Maximum of the normalized 2D cross-correlation between two z-scored
        panels, over the restricted lag window.
        """
        h, w = a.shape

        # Full cross-correlation via FFT convolution with the reversed panel
        cc = fftconvolve(a, b[::-1, ::-1], mode="full")

        # Per-lag overlap count, used to normalize each lag to a Pearson-like value
        ones = np.ones_like(a)
        overlap = fftconvolve(ones, ones, mode="full")
        ncc = cc / np.maximum(overlap, 1.0)

        # Restrict the lag window around zero lag
        ct, cf = h - 1, w - 1
        dt = int(h * self._max_time_lag_frac)
        df = int(w * self._max_freq_lag_frac)
        win = ncc[ct - dt: ct + dt + 1, cf - df: cf + df + 1]

        val = float(win.max())
        return val if np.isfinite(val) else 0.0

    def extract_features(self, cadence: np.ndarray) -> np.ndarray:
        """
        Compute the 15-dimensional cross-correlation feature vector from a cadence.

        Returns
        -------
        np.ndarray
            A vector of length 15 containing, for each unique panel pair
            (upper triangle, k=1, of the 6x6 pair matrix), the maximum of the
            normalized cross-correlation over the restricted lag window.
        """
        cadence = np.asarray(cadence, dtype=np.float64)
        n_panels, _, _ = self._validate_cadence_shape(cadence)

        # Z-scored panels, shape preserved: (6, H, W)
        Z = [self._zscore_panel(cadence[i]) for i in range(n_panels)]

        features = []
        for i in range(n_panels):
            for j in range(i + 1, n_panels):
                features.append(self._ncc_max(Z[i], Z[j]))

        features_15d = np.asarray(features, dtype=np.float64)

        # Sanity check: 6 choose 2 = 15
        if features_15d.shape[0] != 15:
            raise RuntimeError(f"Expected 15 CC features, got {features_15d.shape[0]}")

        # Ensure finite outputs
        if not np.all(np.isfinite(features_15d)):
            features_15d = np.nan_to_num(features_15d, nan=0.0, posinf=0.0, neginf=0.0)

        return features_15d