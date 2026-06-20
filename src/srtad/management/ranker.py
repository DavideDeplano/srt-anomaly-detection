from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import math
import random

from ..core.candidate import Candidate
from ..config import random_seed, ranker


class Ranker:
    """
    Ranking step following the paper's sampling logic.

    Paper logic (conceptual):
    - Top percentile by Frequency score per (target, band)
    - Top percentile by Similarity score per (target, band)
    - Top-K candidates considering both scores equally (combined)
    - Random control sample

    Notes
    -----
    - In current dataset, target metadata is not available, so candidates
      are expected to have target set to a constant placeholder ("UNKNOWN").
      This collapses (target, band) stratification to band-only in practice.
    - Candidates are expected to have:
        - frequency_score (float in [0,1])
        - similarity_score (float in [0,1])
        - band (str) already set ("C", "K")
        - target (str) already set ("UNKNOWN")
    """

    def __init__(self) -> None:
        self._seed = int(random_seed)

        self._top_k = int(ranker["top_k"])
        self._control_k = int(ranker["control_k"])
        self._top_percentile_freq = float(ranker["top_percentile_freq"])
        self._top_percentile_sim = float(ranker["top_percentile_sim"])

    @staticmethod
    def _require_scores(c: Candidate) -> None:
        if c.frequency_score is None:
            raise ValueError(f"Candidate {c.id} missing frequency_score")
        if c.similarity_score is None:
            raise ValueError(f"Candidate {c.id} missing similarity_score")

    @staticmethod
    def _require_meta(c: Candidate) -> None:
        """
        Require band and target metadata for paper-style stratification.
        """
        if getattr(c, "band", None) is None:
            raise ValueError(f"Candidate {c.id} missing band (set it in FrequencyFilter.calculate)")
        if getattr(c, "target", None) is None:
            raise ValueError(f"Candidate {c.id} missing target (set it in Dataset.load via set_target)")

    @staticmethod
    def _group_key(c: Candidate) -> Tuple[str, str]:
        """
        Grouping key: (target, band).
        """
        target = str(getattr(c, "target"))
        band = str(getattr(c, "band"))
        return target, band

    @staticmethod
    def _k_from_percentile(n: int, p: float) -> int:
        """
        Convert a top-percentile into a count.
        For small n, we still return at least 1.
        """
        if n <= 0:
            return 0
        p = max(0.0, min(p, 1.0))
        k = int(math.ceil(p * n))
        return max(1, k)

    @staticmethod
    def _combined_score_equal(c: Candidate) -> float:
        """
        Consider both scores equally — geometric mean: high only if BOTH
        frequency and similarity are high.
        """
        f = max(float(c.frequency_score), 0.0)
        s = max(float(c.similarity_score), 0.0)
        return math.sqrt(f * s)

    def _group_by_target_band(self, candidates: Sequence[Candidate]) -> Dict[Tuple[str, str], List[Candidate]]:
        grouped: Dict[Tuple[str, str], List[Candidate]] = {}
        for c in candidates:
            self._require_meta(c)
            grouped.setdefault(self._group_key(c), []).append(c)
        return grouped

    def top_percentile_per_group(
        self,
        candidates: Sequence[Candidate],
        score_attr: str,
        percentile: float,
    ) -> List[Candidate]:
        """
        Select top percentile by one score independently per (target, band).

        score_attr must be "frequency_score" or "similarity_score".
        """
        if score_attr not in ("frequency_score", "similarity_score"):
            raise ValueError("score_attr must be 'frequency_score' or 'similarity_score'")

        grouped = self._group_by_target_band(candidates)
        selected: List[Candidate] = []

        for _, group in grouped.items():
            valid = [c for c in group if getattr(c, score_attr, None) is not None]
            if not valid:
                continue

            k = self._k_from_percentile(len(valid), percentile)
            valid.sort(key=lambda x: float(getattr(x, score_attr)), reverse=True)
            selected.extend(valid[:k])

        # Global ordering by that score (descending)
        selected.sort(key=lambda x: float(getattr(x, score_attr)), reverse=True)
        return selected

    def top_k_combined(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """
        Top-K considering both scores equally (global list, not stratified).
        """
        valid: List[Candidate] = []
        for c in candidates:
            if c.frequency_score is None or c.similarity_score is None:
                continue
            valid.append(c)

        valid.sort(key=self._combined_score_equal, reverse=True)
        k = min(self._top_k, len(valid))
        return valid[:k]

    def random_control(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """
        Random control sample (global, independent of scores).
        """
        if not candidates:
            return []
        rng = random.Random(self._seed)
        k = min(self._control_k, len(candidates))
        return rng.sample(list(candidates), k=k)

    def build_samples(self, candidates: Sequence[Candidate]) -> Dict[str, List[Candidate]]:
        """
        Build the 4 sets.
        """
        # Optional strict validation (will fail fast if band/target not present)
        for c in candidates:
            self._require_scores(c)
            self._require_meta(c)

        return {
            "top_freq_percentile": self.top_percentile_per_group(
                candidates, "frequency_score", self._top_percentile_freq
            ),
            "top_sim_percentile": self.top_percentile_per_group(
                candidates, "similarity_score", self._top_percentile_sim
            ),
            "top_k_combined": self.top_k_combined(candidates),
            "random_control": self.random_control(candidates),
        }
