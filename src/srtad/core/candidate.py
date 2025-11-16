"""
Candidate entity definition.
Represents a single signal candidate extracted from SRT observations.
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
        # --- Candidate data ---
        self._id = id
        self._frequency_hz = frequency_hz
        self._drift_hz_s = drift_hz_s
        self._cadence = cadence
        self._source_path = source_path

        # --- Scores from filters ---
        # PAPER-ALIGNED: density, frequency, similarity
        self._log_density: float | None = None
        self._frequency_score: float | None = None
        self._similarity_score: float | None = None

    # ========================
    # Properties
    # ========================

    @property
    def id(self) -> str:
        return self._id

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def drift_hz_s(self) -> float:
        return self._drift_hz_s

    @property
    def cadence(self) -> np.ndarray:
        return self._cadence

    @property
    def source_path(self) -> Path:
        return self._source_path

    # ========================
    # Score handling
    # ========================

    @property
    def density_log_density(self) -> float | None:
        return self._log_density

    def set_log_density(self, value: float) -> None:
        """Assign the density score (UMAP+KDE-based)."""
        self._log_density = float(value)

    @property
    def frequency_score(self) -> float | None:
        return self._frequency_score

    def set_frequency_score(self, value: float) -> None:
        """Assign the frequency (GMM-based) score."""
        self._frequency_score = float(value)

    @property
    def similarity_score(self) -> float | None:
        return self._similarity_score

    def set_similarity_score(self, value: float) -> None:
        """Assign the ON/OFF similarity score."""
        self._similarity_score = float(value)

    # ========================
    # Summary
    # ========================

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
