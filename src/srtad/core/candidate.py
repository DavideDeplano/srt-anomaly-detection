"""
Candidate entity definition.
Represents a single signal candidate extracted from SRT observations.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np

class Candidate:
    """
    Parameters
    id : str
        Unique identifier for the candidate (e.g., filename or label).
    frequency_hz : float
        Central frequency of the detected signal in Hertz.
    drift_hz_s : float
        Drift rate of the signal in Hertz per second (Hz/s).
    on_panels : List[np.ndarray]
        Waterfall plots from ON-target observations.
    off_panels : List[np.ndarray]
        Waterfall plots from OFF-target observations.
    snr : Optional[float], default=None
        Signal-to-noise ratio if available.
    source_path : Optional[Path], default=None
        Path to the source file or directory for traceability.
    """

    def __init__(
        self,
        id: str,
        frequency_hz: float,
        drift_hz_s: float,
        on_panels: List[np.ndarray],
        off_panels: List[np.ndarray],
        snr: Optional[float] = None,
        source_path: Path = None,
    ):
       
        # --- Candidate data ---
        self.id = id
        self.frequency_hz = frequency_hz
        self.drift_hz_s = drift_hz_s
        self.on_panels = on_panels
        self.off_panels = off_panels
        self.snr = snr
        self.source_path = source_path

        # --- Scores from calculators ---
        self.cross_correlation_score: Optional[float] = None
        self.frequency_score: Optional[float] = None
        self.similarity_score: Optional[float] = None

        # --- Final ranking results ---
        self.final_rank: Optional[int] = None
        self.final_score: Optional[float] = None

    def to_summary(self) -> Dict[str, Any]:
        """Return a lightweight summary with metadata and computed scores."""
        return {
            "id": self.id,
            "frequency_hz": self.frequency_hz,
            "drift_hz_s": self.drift_hz_s,
            "cross_correlation_score": self.cross_correlation_score,
            "frequency_score": self.frequency_score,
            "similarity_score": self.similarity_score,
            "final_score": self.final_score,
            "final_rank": self.final_rank,
            "source_path": str(self.source_path),
        }

    def set_cross_correlation_score(self, value: float) -> None:
        """Set the cross correlation (UMAP + KDE) score."""
        self.cross_correlation_score = float(value)

    def set_frequency_score(self, value: float) -> None:
        """Set the frequency (GMM) score."""
        self.frequency_score = float(value)

    def set_similarity_score(self, value: float) -> None:
        """Set the ON/OFF similarity score."""
        self.similarity_score = float(value)

    def set_final_score(self, score: float, rank: int) -> None:
        """Assign the final aggregated score and ranking."""
        self.final_score = float(score)
        self.final_rank = int(rank)
