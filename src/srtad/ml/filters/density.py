from typing import Iterable, Dict, Any, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from umap import UMAP
import logging
import math
import joblib
import numpy as np

from src.srtad.ml.filters.i_filter import IFilter
from src.srtad.core.candidate import Candidate
from src.srtad.management.cross_correlation_extractor import CrossCorrelationExtractor
from src.srtad.config import filters, simulation
from src.srtad.management.visualizer import Visualizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _extract_one(
    cadence_id: str,
    tensor: np.ndarray,
    meta: Dict[str, Any],
) -> Tuple[np.ndarray, int] | None:
    """
    Extract 15D CC features and category label from a single synthetic cadence.

    Module-level function required for joblib multiprocessing (pickling).

    Returns
    -------
    (features, category) on success, None on failure.
    """
    try:
        tensor = np.asarray(tensor, dtype=float)
        if tensor.ndim != 3 or tensor.shape[0] != 6:
            return None
        feats = CrossCorrelationExtractor().extract_features(tensor)
        cat = int(meta["pattern_id"])
        return feats, cat
    except Exception:
        return None


class DensityFilter(IFilter):
    """
    Density-based filter using UMAP + KDE on 15D cross-correlation (CC) features.

    Summary
    -------
    - Each cadence (6 panels) is converted into a 15D feature vector using Pearson
      correlations between flattened panels (see CrossCorrelationExtractor).
    - UMAP maps the 15D space into a low-dimensional embedding.
    - One KDE is fitted per simulation category in the UMAP space.
    - During scoring, the candidate is evaluated under all category KDEs and the
      maximum-probability category is selected (argmax over categories).
    - The returned score is the probability only when the argmax category equals
      the configured "only-on" category; otherwise the score is 0.0.
    """

    name: str = "density"

    def __init__(self):
        super().__init__()

        density_cfg = filters["density"]

        self._logger = logging.getLogger("srtad.density")
        self._use_tqdm = True

        self._umap_model_path = Path(density_cfg["umap_model_path"])
        self._kde_models_dir = Path(density_cfg["kde_models_dir"])

        self._cache_Z_train = self._kde_models_dir / "umap_train_points.npy"
        self._cache_y_train = self._kde_models_dir / "umap_train_labels.npy"
        self._cache_Z_val = self._kde_models_dir / "umap_val_points.npy"
        self._cache_y_val = self._kde_models_dir / "umap_val_labels.npy"
        self._cache_probs = self._kde_models_dir / "val_probs.npy"

        self._n_neighbors = int(density_cfg["n_neighbors"])
        self._min_dist = float(density_cfg["min_dist"])
        self._metric = str(density_cfg["umap_metric"])
        self._n_components = int(density_cfg["n_components"])

        self._umap = UMAP(
            n_neighbors=self._n_neighbors,
            min_dist=self._min_dist,
            n_components=self._n_components,
            metric=self._metric,
            random_state=self.random_state,
            verbose=False,
        )

        self._kde_bandwidth = float(density_cfg["kde_bandwidth"])
        self._kde_kernel = str(density_cfg["kernel"])

        self._threshold = float(density_cfg["threshold"])
        self._only_on_category = int(density_cfg["only_on_category"])
        self._only_off_category = int(density_cfg["only_off_category"])

        self._feature_extractor = CrossCorrelationExtractor()

        self._n_categories: int = int(simulation["mask_combinations"])

        self._kdes: Dict[int, KernelDensity] = {}

        self._Z_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._Z_val: np.ndarray | None = None
        self._y_val: np.ndarray | None = None

    def fit(
        self,
        simulated_cadences: Iterable[Tuple[str, np.ndarray, Dict[str, Any]]],
    ) -> None:
        """
        Fit UMAP and per-category KDE models from synthetic cadences.

        CC feature extraction is parallelized across all available CPU cores
        using joblib multiprocessing for significant speedup on large datasets.
        """
        umap_path = self._umap_model_path
        kde_path = self._kde_models_dir / "kdes.joblib"
        meta_path = self._kde_models_dir / "meta.joblib"

        models_exist = umap_path.exists() and kde_path.exists() and meta_path.exists()

        if models_exist:
            self._logger.info("[DENSITY FILTER] Existing models found; loading.")
            self._load_models()
            self._logger.info("[DENSITY FILTER] Models already trained; skipping training step.")
            return

        # Materialise the iterable once so joblib can distribute work.
        cadence_list = list(simulated_cadences)

        try:
            n_sets = int(simulation["n_panel_sets"])
            n_masks = int(simulation["mask_combinations"])
            total_files = n_sets * n_masks
        except (KeyError, ValueError):
            total_files = len(cadence_list)

        print(f"[DENSITY FILTER] Extracting CC features from {len(cadence_list)} cadences "
              f"(parallel, n_jobs=-1)...")

        # Parallel CC feature extraction (CPU-bound → prefer processes).
        raw_results = joblib.Parallel(n_jobs=-1, prefer="processes")(
            joblib.delayed(_extract_one)(cid, tensor, meta)
            for cid, tensor, meta in cadence_list
        )

        X: list[np.ndarray] = []
        y: list[int] = []
        for result in raw_results:
            if result is not None:
                feats, cat = result
                X.append(feats)
                y.append(cat)

        n_skipped = len(cadence_list) - len(X)
        if n_skipped > 0:
            self._logger.warning(
                "[DENSITY FILTER] %d/%d cadences skipped during feature extraction.",
                n_skipped, len(cadence_list),
            )

        if not X:
            raise RuntimeError(
                "DensityFilter.fit: no valid synthetic cadences; cannot train."
            )

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)

        counts = np.bincount(y_arr, minlength=self._n_categories)
        if np.any(counts == 0):
            missing = np.where(counts == 0)[0].tolist()
            raise RuntimeError(f"[DENSITY FILTER] Missing categories in training data: {missing}")

        X_train, X_val, y_train, y_val = train_test_split(
            X_arr,
            y_arr,
            test_size=0.5,
            random_state=self.random_state if self.random_state is not None else 42,
            shuffle=True,
            stratify=y_arr,
        )

        print(f"[DENSITY FILTER] Fitting UMAP on {len(X_train)} training samples...")
        Z_train = self._umap.fit_transform(X_train)
        Z_val = self._umap.transform(X_val)

        print("[DENSITY FILTER] Fitting KDEs...")
        self._kdes.clear()
        for k in range(self._n_categories):
            mask = y_train == k
            if not np.any(mask):
                continue
            kde = KernelDensity(kernel=self._kde_kernel, bandwidth=self._kde_bandwidth)
            kde.fit(Z_train[mask])
            self._kdes[k] = kde

        missing = [k for k in range(self._n_categories) if k not in self._kdes]
        if missing:
            self._logger.warning(
                "[DENSITY FILTER] Missing KDEs for categories: %s (argmax will be biased)", missing
            )

        if self._only_on_category not in self._kdes:
            raise RuntimeError(
                f"Only-on category {self._only_on_category} missing; cannot define score."
            )

        self._validate_on_val(Z_val, y_val)
        self._save_models()
        self._logger.info("[DENSITY FILTER] Training complete.")

        self._kde_models_dir.mkdir(parents=True, exist_ok=True)

        np.save(self._kde_models_dir / "umap_train_points.npy", Z_train)
        np.save(self._kde_models_dir / "umap_train_labels.npy", y_train)
        np.save(self._kde_models_dir / "umap_val_points.npy", Z_val)
        np.save(self._kde_models_dir / "umap_val_labels.npy", y_val)

        only_on_kde = self._kdes[self._only_on_category]
        log_densities = only_on_kde.score_samples(Z_val)
        probs_val = np.exp(log_densities)
        np.save(self._kde_models_dir / "val_probs.npy", probs_val)

        viz = Visualizer()
        viz.plot_umap_embedding(
            Z_val, y_val,
            title="UMAP Validation with KDE Contours",
            filename="umap_paper_figure.png",
            on_cat=self._only_on_category,
            off_cat=self._only_off_category,
            kde_on=self._kdes[self._only_on_category],
            kde_off=self._kdes[self._only_off_category],
        )

    def _derive_category(self, meta: Dict[str, Any]) -> int:
        """Derive the category index from simulation metadata."""
        return int(meta["pattern_id"])

    def calculate(self, candidate: Candidate) -> float:
        """
        Compute the density score for a real candidate.

        Steps
        -----
        1) Ensure models are loaded (UMAP + KDEs).
        2) Validate candidate.cadence has shape (6, H, W).
        3) Extract 15D CC features.
        4) Project to UMAP space.
        5) Score all categories using the corresponding KDEs.
        6) Select best category by maximum log-density.
        7) Return probability only if best category equals only-on; otherwise 0.0.
        """
        if not self._kdes:
            try:
                self._load_models()
            except FileNotFoundError:
                raise RuntimeError(
                    "DensityFilter.calculate: model files not found; run fit() first."
                )

        cadence = getattr(candidate, "cadence", None)
        if cadence is None:
            self._logger.debug(
                "[DENSITY FILTER] Candidate %s has no 'cadence' attribute; returning 0.0.",
                getattr(candidate, "id", "<no-id>"),
            )
            return 0.0

        cadence = np.asarray(cadence, dtype=float)
        if cadence.ndim != 3 or cadence.shape[0] != 6:
            self._logger.debug(
                "[DENSITY FILTER] Candidate %s has invalid cadence shape %s; returning 0.0.",
                getattr(candidate, "id", "<no-id>"),
                cadence.shape,
            )
            return 0.0

        try:
            feats = self._feature_extractor.extract_features(cadence)
        except Exception as exc:
            self._logger.debug(
                "[DENSITY FILTER] CC feature extraction failed for candidate %s: %s. Returning 0.0.",
                getattr(candidate, "id", "<no-id>"),
                repr(exc),
            )
            return 0.0

        z = self._umap.transform(feats.reshape(1, -1))

        log_dens: Dict[int, float] = {}
        for cat in range(self._n_categories):
            kde = self._kdes.get(cat)
            if kde is None:
                log_dens[cat] = float("-inf")
                continue
            try:
                lp = float(kde.score_samples(z)[0])
            except Exception as exc:
                self._logger.debug(
                    "[DENSITY FILTER] KDE scoring failed for candidate %s, category %d: %s",
                    getattr(candidate, "id", "<no-id>"),
                    cat,
                    repr(exc),
                )
                log_dens[cat] = float("-inf")
                continue
            log_dens[cat] = lp if math.isfinite(lp) else float("-inf")

        best_cat = max(log_dens, key=log_dens.get)
        best_logp = log_dens[best_cat]
        best_prob = math.exp(best_logp) if best_logp > float("-inf") else 0.0

        candidate.set_category(best_cat)

        if best_cat != self._only_on_category:
            return 0.0

        return float(best_prob)

    def _validate_on_val(self, Z_val: np.ndarray, y_val: np.ndarray) -> None:
        """Run basic validation diagnostics on a validation embedding."""
        if Z_val.size == 0:
            self._logger.warning(
                "[DENSITY FILTER] Validation set is empty; skipping diagnostics."
            )
            return

        y_val = np.asarray(y_val, dtype=int)

        stats: Dict[int, Tuple[float, float]] = {}
        for k, kde in self._kdes.items():
            mask = y_val == k
            if not np.any(mask):
                continue
            ld = kde.score_samples(Z_val[mask])
            stats[k] = (float(ld.mean()), float(ld.std()))

        for k, (mu, sigma) in stats.items():
            self._logger.info(
                "[DENSITY FILTER] Validation KDE cat=%d: mean log p = %.3f, std = %.3f",
                k, mu, sigma,
            )

        if self._only_on_category in stats:
            mu_only_on = stats[self._only_on_category][0]
            self._logger.info(
                "[DENSITY FILTER] Only-on category (%d): validation mean log p = %.3f",
                self._only_on_category,
                mu_only_on,
            )

    def _save_models(self) -> None:
        """Save UMAP model, KDE models, and metadata to disk."""
        umap_path = self._umap_model_path
        umap_path.parent.mkdir(parents=True, exist_ok=True)

        kde_path = self._kde_models_dir / "kdes.joblib"
        kde_path.parent.mkdir(parents=True, exist_ok=True)

        meta_path = self._kde_models_dir / "meta.joblib"

        joblib.dump(self._umap, umap_path)
        joblib.dump(self._kdes, kde_path)
        joblib.dump(
            {
                "n_categories": self._n_categories,
                "only_on_category": self._only_on_category,
                "kde_kernel": self._kde_kernel,
                "kde_bandwidth": self._kde_bandwidth,
            },
            meta_path,
        )

        self._logger.info(
            "[DENSITY FILTER] Saved models (UMAP=%s, KDEs=%s, meta=%s).",
            umap_path, kde_path, meta_path,
        )

    def _load_models(self) -> None:
        """Load UMAP model, KDE models, and metadata from disk."""
        umap_path = self._umap_model_path
        kde_path = self._kde_models_dir / "kdes.joblib"
        meta_path = self._kde_models_dir / "meta.joblib"

        if not umap_path.exists():
            raise FileNotFoundError(
                f"[DENSITY FILTER] UMAP model file not found: {umap_path}"
            )
        if not self._kde_models_dir.exists():
            raise FileNotFoundError(
                f"[DENSITY FILTER] KDE models directory does not exist: {self._kde_models_dir}"
            )
        if not kde_path.exists():
            raise FileNotFoundError(
                f"[DENSITY FILTER] KDE models file not found: {kde_path}"
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"[DENSITY FILTER] Metadata file not found: {meta_path}"
            )

        self._umap = joblib.load(umap_path)
        self._kdes = joblib.load(kde_path)
        meta = joblib.load(meta_path)

        self._n_categories = int(meta["n_categories"])
        self._only_on_category = int(meta["only_on_category"])
        self._kde_kernel = meta["kde_kernel"]
        self._kde_bandwidth = float(meta["kde_bandwidth"])

        self._logger.info(
            "[DENSITY FILTER] Loaded models (UMAP=%s, KDEs=%s, meta=%s; "
            "n_categories=%d, only_on=%d).",
            umap_path, kde_path, meta_path,
            self._n_categories, self._only_on_category,
        )

        train_points_path = self._kde_models_dir / "umap_train_points.npy"
        train_labels_path = self._kde_models_dir / "umap_train_labels.npy"
        val_points_path = self._kde_models_dir / "umap_val_points.npy"
        val_labels_path = self._kde_models_dir / "umap_val_labels.npy"

        if train_points_path.exists() and train_labels_path.exists():
            self._Z_train = np.load(train_points_path)
            self._y_train = np.load(train_labels_path)
            self._logger.info("Loaded %d training points", len(self._Z_train))

        if val_points_path.exists() and val_labels_path.exists():
            self._Z_val = np.load(val_points_path)
            self._y_val = np.load(val_labels_path)
            self._logger.info("Loaded %d validation points", len(self._Z_val))

    def calculate_with_details(self, candidate) -> tuple:
        """
        Like calculate(), but returns (score, p_only_on, best_cat).

        - score: same as calculate() (prob if best_cat==only_on, else 0.0)
        - p_only_on: probability under the only-on KDE regardless of argmax
        - best_cat: the argmax category
        """
        if not self._kdes:
            try:
                self._load_models()
            except FileNotFoundError:
                raise RuntimeError(
                    "DensityFilter.calculate_with_details: model files not found."
                )

        cadence = getattr(candidate, "cadence", None)
        if cadence is None:
            return 0.0, 0.0, -1

        cadence = np.asarray(cadence, dtype=float)
        if cadence.ndim != 3 or cadence.shape[0] != 6:
            return 0.0, 0.0, -1

        try:
            feats = self._feature_extractor.extract_features(cadence)
        except Exception:
            return 0.0, 0.0, -1

        z = self._umap.transform(feats.reshape(1, -1))

        log_dens = {}
        for cat in range(self._n_categories):
            kde = self._kdes.get(cat)
            if kde is None:
                log_dens[cat] = float("-inf")
                continue
            try:
                lp = float(kde.score_samples(z)[0])
            except Exception:
                log_dens[cat] = float("-inf")
                continue
            log_dens[cat] = lp if math.isfinite(lp) else float("-inf")

        best_cat = max(log_dens, key=log_dens.get)
        best_logp = log_dens[best_cat]
        best_prob = math.exp(best_logp) if best_logp > float("-inf") else 0.0

        logp_on = log_dens.get(self._only_on_category, float("-inf"))
        p_only_on = math.exp(logp_on) if logp_on > float("-inf") else 0.0

        candidate.set_category(best_cat)

        score = float(best_prob) if best_cat == self._only_on_category else 0.0

        return score, p_only_on, best_cat