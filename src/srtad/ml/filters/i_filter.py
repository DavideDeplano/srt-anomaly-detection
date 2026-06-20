"""
IFilter interface.

Defines the base contract for all scoring modules in the 
anomaly detection pipeline.
"""

from abc import ABC, abstractmethod
from typing import Iterable
from ...config import random_seed
from ...core.candidate import Candidate


class IFilter(ABC):
    """
    Abstract base class for all filter/scoring components.

    This interface defines a common API that all concrete filters
    (e.g. density-based, frequency-based, similarity-based) must implement.

    Responsibilities
    ----------------
    - Expose a consistent `fit()` / `calculate()` interface.
    - Provide access to a shared random state for reproducibility.
    - Define hooks for model persistence (load/save).
    """

    # Identifier for the filter (to be overridden by subclasses)
    name: str = "base"

    def __init__(self) -> None:
        """
        Initialize shared configuration for the filter.

        Notes
        -----
        - The random seed is read from the global configuration and stored
          as `random_state` for reproducibility.
        """
        self._random_state = int(random_seed)

    @property
    def random_state(self) -> int:
        """
        Random state associated with this filter.

        Intended to be used by subclasses when initializing stochastic
        models or sampling procedures.
        """
        return self._random_state

    @abstractmethod
    def fit(self, candidates: Iterable[Candidate]) -> None:
        """
        Fit the filter on a collection of candidates.

        This method is used by filters that require dataset-level
        statistics or model training (e.g. KDE, GMM).
        """
        raise NotImplementedError

    @abstractmethod
    def calculate(self, candidate: Candidate) -> float:
        """
        Compute the raw score for a single candidate.

        Returns
        -------
        float
            A finite scalar score. Higher values typically indicate
            a more anomalous or interesting candidate.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_models(self) -> None:
        """
        Load any previously saved model parameters from disk.

        The concrete storage format and location are defined
        by the implementing subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _save_models(self) -> None:
        """
        Save model parameters to disk.

        The concrete storage format and location are defined
        by the implementing subclass.
        """
        raise NotImplementedError
