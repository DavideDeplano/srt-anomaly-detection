from typing import Iterable, List, Dict
import logging
import math
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
import joblib

from .i_filter import IFilter
from ...core.candidate import Candidate
from ...config import filters

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

class FrequencyFilter(IFilter):
    """
    Frequency-based filter using Gaussian Mixture Models (GMM) with bagging.

    Purpose
    -------
    Estimate how "rare" a candidate frequency is within its observing band
    (L, S, 10cm) by learning a probabilistic density model from training data

    Model
    -----
    - For each band, train an ensemble of GMMs (bagging)
    - Each GMM is fitted on a random subset of candidate frequencies in that band
    - At inference time, evaluate the ensemble mean PDF at the candidate frequency

    Scoring convention
    ------------------
    Higher score means more anomalous (rarer frequency under the trained density)

    Raw score definition
    --------------------
    raw_score = 1 / mean_pdf
    where mean_pdf is the average (across the bagged GMMs) of the PDF value
    at the candidate frequency

    Normalization
    -------------
    A global min/max range of raw scores is computed on training candidates and
    used to linearly scale scores into [0, 1] during inference when possible
    """

    # Filter identifier used by the pipeline
    name: str = "frequency"

    def __init__(self) -> None:
        """
        Initialize the FrequencyFilter from YAML configuration

        Configuration keys (filters.frequency)
        -------------------------------------
        - gmm_model_path: output path for the saved ensemble state
        - n_components_L / n_components_S / n_components_10cm: default mixtures per band
        - n_bags: number of GMMs to train per band
        - bag_size: number of samples per bag (subsample size)
        """
        super().__init__()

        frequency_cfg = filters["frequency"]

        self._logger = logging.getLogger("srtad.frequency")
        self._use_tqdm = True

        # Single persistence file containing ensembles + scaling stats
        self._gmm_model_path = Path(frequency_cfg["gmm_model_path"])

        # Default number of components for each band GMM
        self._components: Dict[str, int] = {
            "L": int(frequency_cfg["n_components_L"]),
            "S": int(frequency_cfg["n_components_S"]),
            "10cm": int(frequency_cfg["n_components_10cm"]),
        }

        # Bagging parameters
        self._n_bags = int(frequency_cfg["n_bags"])
        self._bag_size = int(frequency_cfg["bag_size"])

        # Ensembles: band -> list of fitted GaussianMixture models
        self._gmm_ensembles: Dict[str, List[GaussianMixture]] = {
            "L": [],
            "S": [],
            "10cm": [],
        }

        # Global min/max used to scale raw scores into [0, 1]
        self._raw_min: float = 0.0
        self._raw_max: float = 0.0

    @staticmethod
    def _extract_band(candidate: Candidate) -> str | None:
        """
        Assign a band label based on the candidate central frequency

        Band definitions are intentionally overlapping
        ---------------------------------------------
        - L band:    1.10-1.90 GHz
        - S band:    1.80-2.80 GHz
        - 10cm band: 2.60-3.45 GHz

        Overlap resolution policy
        -------------------------
        Because this function uses an if/elif chain, overlaps are resolved by a
        deterministic first-match-wins rule:
        - 1.80-1.90 GHz is assigned to L (L checked before S)
        - 2.60-2.80 GHz is assigned to S (S checked before 10cm)
        """
        f = candidate.frequency_hz

        if 1.10e9 <= f <= 1.90e9:
            return "L"
        elif 1.80e9 <= f <= 2.80e9:
            return "S"
        elif 2.60e9 <= f <= 3.45e9:
            return "10cm"
        else:
            # Frequency outside the supported band ranges
            return None

    def fit(self, candidates: Iterable[Candidate]) -> None:
        """
        Train the GMM bagging ensembles for each band and persist them to disk

        Steps
        -----
        1) Group candidates by band using _extract_band
        2) For each band, train n_bags GMMs on random subsamples (bagging)
        3) Compute global min/max of raw scores over the training set
        4) Save ensembles and scaling parameters
        """
        self._logger.info(
            "[FREQUENCY FILTER] Training mode: fitting GMM ensembles over frequencies..."
        )

        # Group candidates by band label
        band_to_candidates: Dict[str, List[Candidate]] = {"L": [], "S": [], "10cm": []}
        for c in candidates:
            b = self._extract_band(c)
            if b is not None:
                band_to_candidates[b].append(c)

        # Reproducible RNG for bag sampling
        rng = np.random.default_rng(self.random_state)

        # Fit a bagged ensemble of GMMs per band
        for band_name, band_candidates in band_to_candidates.items():
            n_candidates = len(band_candidates)
            if n_candidates == 0:
                self._logger.warning(
                    "[FREQUENCY FILTER] Band %s: no candidates found; "
                    "skipping GMM training.",
                    band_name,
                )
                continue

            self._logger.info(
                "[FREQUENCY FILTER] Band %s: %d candidates, "
                "n_components=%d, n_bags=%d, bag_size=%d",
                band_name,
                n_candidates,
                self._components[band_name],
                self._n_bags,
                self._bag_size,
            )

            # Training data: frequencies as a column vector (N, 1)
            freqs = np.array(
                [c.frequency_hz for c in band_candidates],
                dtype=float,
            ).reshape(-1, 1)

            iterator = range(self._n_bags)
            if self._use_tqdm and tqdm is not None:
                iterator = tqdm(
                    iterator,
                    desc=f"GMM training ({band_name}-band)",
                    leave=False,
                )

            # Reset ensemble for this band before training
            self._gmm_ensembles[band_name] = []

            for _ in iterator:
                current_bag_size = min(self._bag_size, n_candidates)
                if current_bag_size <= 0:
                    continue

                # Sample without replacement to diversify bags
                idx = rng.choice(n_candidates, size=current_bag_size, replace=False)
                sample = freqs[idx]

                # Sample without replacement to diversify bags
                n_comp = min(self._components[band_name], current_bag_size)
                if n_comp < 1:
                    continue

                # Fit one GMM for this bag
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type="full",
                    random_state=self.random_state,
                )
                gmm.fit(sample)
                self._gmm_ensembles[band_name].append(gmm)

        # Compute global min/max of raw scores for later normalization
        self._compute_scaling(band_to_candidates)

        # Persist ensembles and scaling parameters
        self._save_models()

        self._logger.info("[FREQUENCY FILTER] Fit complete.")

    def _compute_scaling(self, band_to_candidates: Dict[str, List[Candidate]]) -> None:
        """
        Compute global min/max of raw scores across training candidates

        This provides a stable linear mapping into [0, 1] during inference:
            scaled = (raw - raw_min) / (raw_max - raw_min)
        """
        raw_scores: List[float] = []

        for band_name, candidates in band_to_candidates.items():
            if not self._gmm_ensembles.get(band_name):
                continue

            # Collect raw scores over candidates in that band
            for c in candidates:
                raw = self._raw_density_score(c)
                if math.isfinite(raw):
                    raw_scores.append(raw)

        if not raw_scores:
            self._raw_min = 0.0
            self._raw_max = 0.0
            self._logger.warning(
                "[FREQUENCY FILTER] No valid raw scores for scaling; "
                "using degenerate [0, 0] range."
            )
            return

        self._raw_min = float(np.min(raw_scores))
        self._raw_max = float(np.max(raw_scores))

        self._logger.info(
            "[FREQUENCY FILTER] Raw score scaling range: min=%.4f, max=%.4f",
            self._raw_min,
            self._raw_max,
        )

    def _save_models(self) -> None:
        """
        Save GMM ensembles and scaling parameters to disk

        The saved state contains:
        - components: per-band default component counts
        - gmm_ensembles: trained GMM lists per band
        - raw_min/raw_max: global scaling range for normalization
        """
        path = self._gmm_model_path

        if path is None or not str(path):
            self._logger.warning(
                "[FREQUENCY FILTER] gmm_model_path not set; skipping save."
            )
            return

        # Ensure output directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "components": self._components,
            "gmm_ensembles": self._gmm_ensembles,
            "raw_min": self._raw_min,
            "raw_max": self._raw_max,
        }

        self._logger.info(
            "[FREQUENCY FILTER] Saving GMM ensembles to %s", path
        )
        joblib.dump(state, path)

    def _load_models(self) -> None:
        """
        Load GMM ensembles and scaling parameters from disk.
        """
        path = self._gmm_model_path

        if path is None or not str(path):
            msg = (
                "[FREQUENCY FILTER] gmm_model_path is empty in "
                "inference mode."
            )
            self._logger.error(msg)
            raise RuntimeError(msg)

        if not path.exists():
            msg = f"[FREQUENCY FILTER] GMM model file not found: {path}"
            self._logger.error(msg)
            raise FileNotFoundError(msg)

        self._logger.info(
            "[FREQUENCY FILTER] Loading GMM ensembles from %s", path
        )
        state = joblib.load(path)

        # Restore state with safe defaults
        self._components = state.get("components", self._components)
        self._gmm_ensembles = state.get("gmm_ensembles", self._gmm_ensembles)
        self._raw_min = state.get("raw_min", None)
        self._raw_max = state.get("raw_max", None)

        # Sanity check: at least one band must have an ensemble
        if not any(self._gmm_ensembles.values()):
            msg = "[FREQUENCY FILTER] Loaded state has no GMM ensembles."
            self._logger.error(msg)
            raise RuntimeError(msg)

    def calculate(self, candidate: Candidate) -> float:
        """
        Compute the frequency-based anomaly score for a candidate

        Steps
        -----
        1) Ensure ensembles are loaded from disk
        2) Determine candidate band
        3) Evaluate raw anomaly score from the band ensemble (1 / mean_pdf)
        4) Optionally scale raw score into [0, 1] using training min/max

        Returns
        -------
        float
            Scaled score in [0, 1] when scaling is available and non-degenerate
            Otherwise a raw or zero value depending on state validity
        """
        # Auto-load models from disk if ensembles are empty
        if not any(self._gmm_ensembles.values()):
            try:
                self._load_models()
            except FileNotFoundError:
                raise RuntimeError(
                    "[FREQUENCY FILTER] Model file not found; call fit(...) first "
                    "to train and save models."
                )

        # Map frequency to band
        band = self._extract_band(candidate)
        if band is None:
            # Out-of-band frequencies are not scored
            return 0.0

        # If band ensemble is missing, return neutral score
        if not self._gmm_ensembles.get(band):
            return 0.0

        # Compute raw anomaly score (higher means rarer)
        raw = self._raw_density_score(candidate)

        # Normalize using training min/max when available
        if self._raw_min is not None and self._raw_max is not None:
            if self._raw_max > self._raw_min:
                raw = (raw - self._raw_min) / (self._raw_max - self._raw_min)
            else:
                # Degenerate scaling range (avoid division by zero)
                raw = 0.0

        return float(raw)


    def _raw_density_score(self, candidate: Candidate) -> float:
        """
        Compute raw frequency anomaly score for a candidate

        Definition
        ----------
        - Evaluate log-density for each GMM in the band ensemble at x = frequency
        - Convert to PDF values via exp(logp)
        - Compute mean_pdf across the ensemble
        - Return 1 / mean_pdf

        Rationale
        ---------
        Frequencies with low density (rare under training distribution) yield low mean_pdf
        and therefore high anomaly scores
        """
        band = self._extract_band(candidate)
        if band is None or not self._gmm_ensembles.get(band):
            return 0.0

        # Input format required by sklearn: shape (n_samples, n_features)
        x = np.array([[candidate.frequency_hz]], dtype=float)

        # Collect per-model log densities at x
        log_scores = np.array(
            [gmm.score_samples(x)[0] for gmm in self._gmm_ensembles[band]], 
            dtype=float
        )

        # Convert log-density to density
        pdf_scores = np.exp(log_scores)

        # Ensemble mean density
        mean_pdf = float(np.mean(pdf_scores))
        if mean_pdf <= 0.0 or not math.isfinite(mean_pdf):
            return 0.0
        
        return 1.0 / mean_pdf
