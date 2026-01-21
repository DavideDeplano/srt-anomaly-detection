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

        # --- Outputs from pipeline filters ---
        # NOTE: "category" is a diagnostic helper, not a semantic label
        # It stores the argmax category assigned by the FIRST-stage density filter
        # (UMAP + KDE over simulated categories). It is useful for reporting/debugging
        # how candidates distribute across KDE categories, and to explain rejections
        self._category: int | None = None

        # Density score produced by the UMAP + KDE filter (first-stage filtering)
        self._density_score: float | None = None

        # Frequency score produced by the GMM ensemble filter
        self._frequency_score: float | None = None

        # Similarity score quantifying ON/OFF consistency in embedded space
        self._similarity_score: float | None = None

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
        """
        return self._cadence

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

    def to_summary(self) -> Dict[str, Any]:
        """Return a lightweight summary with metadata and computed scores."""
        return {
            "id": self.id,
            "frequency_hz": self.frequency_hz,
            "drift_hz_s": self.drift_hz_s,
            "density_score": self.density_score,
            "frequency_score": self.frequency_score,
            "similarity_score": self.similarity_score,
            "source_path": str(self.source_path),
        }
