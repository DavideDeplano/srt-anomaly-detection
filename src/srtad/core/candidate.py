"""
Candidate entity definition.

Represents a single signal candidate extracted from SRT observations.
This class is intentionally a lightweight data container: it stores
metadata, the cadence tensor, and scores produced by the pipeline filters.
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np

class Candidate:
    """
    Parameters
    ----------
    id : str
        Unique identifier for the candidate (e.g., filename or label).
    frequency_hz : float
        Central frequency of the detected signal in Hertz.
    drift_hz_s : float
        Drift rate of the signal in Hertz per second.
    cadence : np.ndarray
        Full cadence tensor with shape (6, H, W).
    source_path : Path
        Path to the original PNG file for traceability.
    """

    def __init__(
        self,
        id: str,
        frequency_hz: float,
        drift_hz_s: float,
        cadence: np.ndarray,
        source_path: Path
    ):
        # --- Core candidate metadata ---
        self._id = id
        self._frequency_hz = frequency_hz
        self._drift_hz_s = drift_hz_s
        self._cadence = cadence
        self._source_path = source_path

        # Min-max (non-bandpassed) cadence used by the similarity filter.
        # The density/frequency filters use the bandpassed `cadence`; the
        # bandpass correction removes persistent narrowband lines, so the
        # similarity filter instead works on this min-max representation.
        self._cadence_raw: np.ndarray | None = None

        # --- Outputs from pipeline filters ---
        # NOTE: "category" is a diagnostic helper, not a semantic label
        # It stores the argmax category assigned by the FIRST-stage density filter
        # (UMAP + KDE over simulated categories).
        self._category: int | None = None

        # Density score produced by the UMAP + KDE filter (first-stage filtering)
        self._density_score: float | None = None

        # Frequency score produced by the GMM ensemble filter
        self._frequency_score: float | None = None

        # Similarity score quantifying ON/OFF consistency in embedded space
        self._similarity_score: float | None = None

        # Source target identifier used for stratification in ranking
        # NOTE: not aviable in the current dataset
        self._target: str | None = None

        # Band information extracted from the candidate's frequency, used for band-specific ranking
        self._band: str | None = None

        # Anomaly score produced by the AE filter
        self._ae_score: float | None = None

    @property
    def id(self) -> str:
        """Unique identifier for this candidate."""
        return self._id

    @property
    def frequency_hz(self) -> float:
        """Central frequency (Hz)."""
        return self._frequency_hz

    @property
    def drift_hz_s(self) -> float:
        """Drift rate (Hz/s)."""
        return self._drift_hz_s

    @property
    def cadence(self) -> np.ndarray:
        """
        Full cadence tensor with shape (6, H, W).

        The 6 panels represent the ON/OFF observation pattern used in the pipeline.
        This is the bandpassed (Pardo-style) representation used by the density
        and frequency filters.
        """
        return self._cadence

    @property
    def cadence_raw(self) -> np.ndarray | None:
        """
        Min-max (non-bandpassed) cadence tensor with shape (6, H, W).

        Used by the similarity filter. The bandpass correction applied to
        `cadence` removes persistent narrowband signals; this representation
        preserves them, so the similarity filter scores on it instead.
        """
        return self._cadence_raw

    def set_cadence_raw(self, value: np.ndarray) -> None:
        """Set the min-max (non-bandpassed) cadence for the similarity filter."""
        self._cadence_raw = np.asarray(value, dtype=float)

    @property
    def source_path(self) -> Path:
        """Original file path associated with this candidate (traceability)."""
        return self._source_path

    @property
    def category(self) -> int | None:
        """
        Diagnostic helper category assigned by the density filter.

        This is the argmax category over KDE probabilities in the UMAP space,
        corresponding to the simulated category that best matches the candidate.
        It is NOT a final classification label and should not be interpreted
        as a semantic class.
        """
        return self._category

    def set_category(self, value: int) -> None:
        """Set the diagnostic density argmax category."""
        self._category = int(value)


    @property
    def density_score(self) -> float | None:
        """Density score from the UMAP + KDE filter."""
        return self._density_score

    def set_density_score(self, value: float) -> None:
        """Set the density score from the UMAP + KDE filter."""
        self._density_score = float(value)

    @property
    def frequency_score(self) -> float | None:
        """Frequency-based score from the GMM ensemble filter."""
        return self._frequency_score

    def set_frequency_score(self, value: float) -> None:
        """Set the frequency-based score from the GMM ensemble filter."""
        self._frequency_score = float(value)

    @property
    def similarity_score(self) -> float | None:
        """ON/OFF similarity score."""
        return self._similarity_score

    def set_similarity_score(self, value: float) -> None:
        """Set the ON/OFF similarity score."""
        self._similarity_score = float(value)

    @property
    def target(self) -> str | None:
        """Source target identifier used for stratification in ranking."""
        return self._target
    
    def set_target(self, value: str) -> None:
        """Set the source target identifier used for stratification in ranking."""
        self._target = str(value)

    @property
    def band(self) -> str | None:
        """Band information extracted from the candidate's frequency."""
        return self._band
    
    def set_band(self, value: str) -> None:
        """Set the band information extracted from the candidate's frequency."""
        self._band = str(value)

    @property
    def ae_score(self) -> float | None:
        """Reconstruction error score from the convolutional AutoencoderFilter."""
        return self._ae_score
    
    def set_ae_score(self, value: float) -> None:
        """Set the reconstruction error score from the convolutional AutoencoderFilter."""
        self._ae_score = float(value)

    def to_summary(self) -> Dict[str, Any]:
        """Return a lightweight summary with metadata and computed scores."""
        return {
            "id": self.id,
            "frequency_hz": self.frequency_hz,
            "drift_hz_s": self.drift_hz_s,
            "density_score": self.density_score,
            "frequency_score": self.frequency_score,
            "similarity_score": self.similarity_score,
            "target": self.target,
            "band": self.band,
            "source_path": str(self.source_path),
            "ae_score": self.ae_score
        }