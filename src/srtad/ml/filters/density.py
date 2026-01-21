from typing import Iterable, Dict, Any, Tuple
from pathlib import Path
import logging
import math

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from umap import UMAP

from .i_filter import IFilter
from ...core.candidate import Candidate
from ...management.cross_correlation_extractor import CrossCorrelationExtractor
from ...config import filters, simulation
from ...management.visualizer import Visualizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


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

        # Read density filter configuration from the global config dictionary.
        density_cfg = filters["density"]

        self._logger = logging.getLogger("srtad.density")
        self._use_tqdm = True

        # Model persistence paths (UMAP + KDE collection + metadata).
        self._umap_model_path = Path(density_cfg["umap_model_path"])
        self._kde_models_dir = Path(density_cfg["kde_models_dir"])

        # Optional caches for validation plots/diagnostics.
        self._cache_Z_train = self._kde_models_dir / "umap_train_points.npy"  
        self._cache_y_train = self._kde_models_dir / "umap_train_labels.npy"
        self._cache_Z_val = self._kde_models_dir / "umap_val_points.npy"
        self._cache_y_val = self._kde_models_dir / "umap_val_labels.npy"
        self._cache_probs = self._kde_models_dir / "val_probs.npy"

        # UMAP hyperparameters (read from config).
        self._n_neighbors = int(density_cfg["n_neighbors"])
        self._min_dist = float(density_cfg["min_dist"])
        self._metric = str(density_cfg["umap_metric"])
        self._n_components = int(density_cfg["n_components"])

         # UMAP model instance (fit on training CC features).
        self._umap = UMAP(
            n_neighbors=self._n_neighbors,
            min_dist=self._min_dist,
            n_components=self._n_components,
            metric=self._metric,
            random_state=self.random_state,
            verbose=False
        )

        # KDE hyperparameters (read from config).
        self._kde_bandwidth = float(density_cfg["kde_bandwidth"])
        self._kde_kernel = str(density_cfg["kernel"])

        # Scoring configuration (read from config).
        # - threshold: value used for plotting and external decision logic
        # - only_on/off categories: special category IDs used by the pipeline
        self._threshold = float(density_cfg["threshold"])
        self._only_on_category = int(density_cfg["only_on_category"])
        self._only_off_category = int(density_cfg["only_off_category"])

        # CC feature extractor (cadence -> 15D).
        self._feature_extractor = CrossCorrelationExtractor()

        # Number of categories expected in the simulation scheme (64).
        self._n_categories: int = int(simulation["mask_combinations"])

        # Mapping: category_id -> fitted KernelDensity model.
        self._kdes: Dict[int, KernelDensity] = {}

    def fit(
        self,
        simulated_cadences: Iterable[Tuple[str, np.ndarray, Dict[str, Any]]],
    ) -> None:
        """
        Fit UMAP and per-category KDE models from synthetic cadences.

        Parameters
        ----------
        simulated_cadences:
            Iterable of (cadence_id, tensor, metadata_dict) where:
            - cadence_id: str identifier used to locate the corresponding .npy file
            - tensor: np.ndarray with shape (6, H, W)
            - metadata_dict: must include at least "pattern_id" to derive category

        Behavior
        --------
        - If model files already exist on disk, models are loaded and training is skipped.
        - Otherwise:
            1) Extract 15D CC features for each cadence
            2) Derive category label from metadata
            3) Split into train/validation sets (stratified)
            4) Fit UMAP on training features and transform validation features
            5) Fit one KDE per category in UMAP space
            6) Run validation diagnostics
            7) Save models and optional caches/plots
        """
        umap_path = self._umap_model_path
        kde_path = self._kde_models_dir / "kdes.joblib"
        meta_path = self._kde_models_dir / "meta.joblib"

        # Local variables used only when training from scratch or loading plot caches.
        Z_train, y_train, Z_val, y_val = None, None, None, None
        probs_val = None

        # Model presence check (UMAP + KDEs + meta must all exist).
        models_exist = umap_path.exists() and kde_path.exists() and meta_path.exists()

        # Model presence check (UMAP + KDEs + meta must all exist).
        if models_exist:
            self._logger.info("[DENSITY FILTER] Existing models found; loading.")
            self._load_models()

            # Optional: load cached UMAP points for plots (if available).
            if self._cache_Z_train.exists() and self._cache_Z_val.exists():
                try:
                    Z_train = np.load(self._cache_Z_train)
                    y_train = np.load(self._cache_y_train)
                    Z_val = np.load(self._cache_Z_val)
                    y_val = np.load(self._cache_y_val)
                    if self._cache_probs.exists():
                        probs_val = np.load(self._cache_probs)
                    loaded_from_cache = True
                    self._logger.info("Loaded UMAP points from cache.")
                except Exception as e:
                    self._logger.warning(f"Failed to load UMAP points from cache: {e}")
                    loaded_from_cache = False

        # If models exist, skip training entirely.
        if models_exist:
            self._logger.info("[DENSITY FILTER] Models already trained; skipping training step.")
            return

        # Accumulators for training samples.
        X: list[np.ndarray] = []
        y: list[int] = []

        # Optional: expected total size for progress reporting.
        try:
            n_sets = int(simulation["n_panel_sets"])
            n_masks = int(simulation["mask_combinations"])
            total_files = n_sets * n_masks
        except (KeyError, ValueError):
            total_files = None

        iterator = simulated_cadences
        if self._use_tqdm and tqdm is not None:
            iterator = tqdm(
                simulated_cadences,
                desc="CC feature extraction (simulated cadences)",
                total=total_files
            )

        # Extract features and labels from each synthetic cadence.
        for cadence_id, tensor, meta in iterator:
            # Validate tensor shape (must be (6, H, W)).
            try:
                tensor = np.asarray(tensor, dtype=float)
                if tensor.ndim != 3 or tensor.shape[0] != 6:
                    continue
            except Exception:
                continue

            # Compute 15D CC features.
            try:
                feats = self._feature_extractor.extract_features(tensor)
            except Exception:
                continue

            # Derive category label from metadata.
            try:
                cat = self._derive_category(meta)
            except Exception:
                continue

            X.append(feats)
            y.append(int(cat))

        # Fail fast if nothing usable was extracted.
        if not X:
            raise RuntimeError(
                "DensityFilter.fit: no valid synthetic cadences; cannot train."
            )

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)

        # Require that all categories appear in training data (strict check).
        counts = np.bincount(y_arr, minlength=self._n_categories)
        if np.any(counts == 0):
            missing = np.where(counts == 0)[0].tolist()
            raise RuntimeError(f"[DENSITY FILTER] Missing categories in training data: {missing}")

        # Train/validation split (stratified to preserve category distribution).
        X_train, X_val, y_train, y_val = train_test_split(
            X_arr,
            y_arr,
            test_size=0.5,
            random_state=self.random_state if self.random_state is not None else 42,
            shuffle=True,
            stratify=y_arr
        )

        # Fit UMAP on training set and transform validation set.
        Z_train = self._umap.fit_transform(X_train)
        Z_val = self._umap.transform(X_val)

        # Fit a KDE model per category in the UMAP space.
        self._kdes.clear()
        for k in range(self._n_categories):
            mask = (y_train == k)
            if not np.any(mask):
                continue
            kde = KernelDensity(kernel=self._kde_kernel, bandwidth=self._kde_bandwidth)
            kde.fit(Z_train[mask])
            self._kdes[k] = kde

        # Warn if any KDE is missing (argmax over categories will be affected).
        missing = [k for k in range(self._n_categories) if k not in self._kdes]
        if missing:
            self._logger.warning(
                f"[DENSITY FILTER] Missing KDEs for categories: {missing} "
                "(argmax will be biased)"
            )

        # The only-on category must exist to define the returned score.
        if self._only_on_category not in self._kdes:
            raise RuntimeError(
                f"Only-on category {self._only_on_category} missing; cannot define score."
            )

        # Validation diagnostics on the validation set.
        self._validate_on_val(Z_val, y_val)

        # Save trained models to disk.
        self._save_models()
        self._logger.info("[DENSITY FILTER] Training complete.")

        # Save UMAP embedding points and labels for external plotting/inspection.
        self._logger.info("[DENSITY FILTER] Saving UMAP points to cache...")
        self._kde_models_dir.mkdir(parents=True, exist_ok=True)

        np.save(self._kde_models_dir / "umap_train_points.npy", Z_train)
        np.save(self._kde_models_dir / "umap_train_labels.npy", y_train)
        np.save(self._kde_models_dir / "umap_val_points.npy", Z_val)
        np.save(self._kde_models_dir / "umap_val_labels.npy", y_val)
        
        # Compute only-on KDE probabilities for validation points (exp of log-density).
        only_on_kde = self._kdes[self._only_on_category]
        log_densities = only_on_kde.score_samples(Z_val)
        probs_val = np.exp(log_densities)

        np.save(self._kde_models_dir / "val_probs.npy", probs_val)

        # Produce validation plots using the cached points and the configured threshold.
        viz = Visualizer()

        viz.plot_umap_embedding(
            Z_val, y_val,
            title="UMAP Validation with KDE Contours",
            filename="umap_paper_figure.png",
            on_cat=self._only_on_category,
            off_cat=self._only_off_category,
            kde_on=self._kdes[self._only_on_category],
            kde_off=self._kdes[self._only_off_category]
        )

        viz.plot_density_histogram(
            probs_val,
            threshold=self._threshold,
            filename="density_validation_hist.png"
        )

    def _derive_category(self, meta: Dict[str, Any]) -> int:
        """
        Derive the category index from simulation metadata.

        Current rule:
            category = int(meta["pattern_id"])
        """
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
        6) Select best category by maximum log-density (equivalently max probability).
        7) Return probability only if best category equals the configured only-on category;
           otherwise return 0.0.

        Returns
        -------
        float
            Probability value for the best category (exp of best log-density) if the
            best category is only-on; otherwise 0.0.
        """
        # 1) Auto-load models if needed
        if not self._kdes:
            try:
                self._load_models()
            except FileNotFoundError:
                raise RuntimeError(
                    "DensityFilter.calculate: model files not found; run fit() first."
                )

        # 2) Retrieve cadence tensor from the candidate
        cadence = getattr(candidate, "cadence", None)
        if cadence is None:
            self._logger.debug(
                "[DENSITY FILTER] Candidate %s has no 'cadence' attribute; "
                "returning 0.0.",
                getattr(candidate, "id", "<no-id>"),
            )
            return 0.0

        cadence = np.asarray(cadence, dtype=float)
        if cadence.ndim != 3 or cadence.shape[0] != 6:
            self._logger.debug(
                "[DENSITY FILTER] Candidate %s has invalid cadence shape %s "
                "(expected (6, H, W)); returning 0.0.",
                getattr(candidate, "id", "<no-id>"),
                cadence.shape,
            )
            return 0.0

        # 3) Cross-correlation features (15D)
        try:
            feats = self._feature_extractor.extract_features(cadence)
        except Exception as exc:
            self._logger.debug(
                "[DENSITY FILTER] CC feature extraction failed for candidate %s: "
                "%s. Returning 0.0.",
                getattr(candidate, "id", "<no-id>"),
                repr(exc),
            )
            return 0.0

        # 4) UMAP embedding (2D)
        z = self._umap.transform(feats.reshape(1, -1))  # shape (1, d)

        # 5) Score all categories in log-density space
        # Missing KDEs are treated as -inf log-density (zero probability)
        log_dens: Dict[int, float] = {}

        for cat in range(self._n_categories):
            kde = self._kdes.get(cat)

            # Missing KDE -> treat as zero density
            if kde is None:
                log_dens[cat] = float("-inf")
                continue

            try:
                lp = float(kde.score_samples(z)[0])
            except Exception as exc:
                self._logger.debug(
                    "[DENSITY FILTER] KDE scoring failed for candidate %s, "
                    "category %d: %s",
                    getattr(candidate, "id", "<no-id>"),
                    cat,
                    repr(exc),
                )
                log_dens[cat] = float("-inf")
                continue

            log_dens[cat] = lp if math.isfinite(lp) else float("-inf")

        # 6) Select best category and convert to probability
        best_cat = max(log_dens, key=log_dens.get)
        best_logp = log_dens[best_cat]
        best_prob = math.exp(best_logp) if best_logp > float("-inf") else 0.0

        # Store the argmax category as a diagnostic helper on the candidate
        candidate.set_category(best_cat)

        # 7) Keep only candidates whose argmax category equals the configured only-on category
        if best_cat != self._only_on_category:
            return 0.0

        return float(best_prob)

    def _validate_on_val(self, Z_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Run basic validation diagnostics on a validation embedding.

        For each category present in y_val:
        - compute mean and standard deviation of log-densities under the
          corresponding KDE model.
        - log summary statistics.
        """
        if Z_val.size == 0:
            self._logger.warning(
                "[DENSITY FILTER] Validation set is empty; skipping diagnostics."
            )
            return

        y_val = np.asarray(y_val, dtype=int)

        stats: Dict[int, Tuple[float, float]] = {}
        for k, kde in self._kdes.items():
            mask = (y_val == k)
            if not np.any(mask):
                continue

            ld = kde.score_samples(Z_val[mask])
            stats[k] = (float(ld.mean()), float(ld.std()))

        for k, (mu, sigma) in stats.items():
            self._logger.info(
                "[DENSITY FILTER] Validation KDE cat=%d: mean log p = %.3f, "
                "std = %.3f",
                k,
                mu,
                sigma,
            )

        if self._only_on_category in stats:
            mu_only_on = stats[self._only_on_category][0]
            self._logger.info(
                "[DENSITY FILTER] Only-on category (%d): validation mean log p "
                "= %.3f",
                self._only_on_category,
                mu_only_on,
            )

    def _save_models(self) -> None:
        """
        Save UMAP model, KDE models, and metadata to disk.
        """
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
            umap_path,
            kde_path,
            meta_path,
        )

    def _load_models(self) -> None:
        """
        Load UMAP model, KDE models, and metadata from disk.
        """
        umap_path = self._umap_model_path
        kde_path = self._kde_models_dir / "kdes.joblib"
        meta_path = self._kde_models_dir / "meta.joblib"

        if not umap_path.exists():
            raise FileNotFoundError(
                f"[DENSITY FILTER] UMAP model file not found: {umap_path}"
            )

        if not self._kde_models_dir.exists():
            raise FileNotFoundError(
                f"[DENSITY FILTER] KDE models directory does not exist: "
                f"{self._kde_models_dir}"
            )

        if not kde_path.exists():
            raise FileNotFoundError(
                f"[DENSITY FILTER] KDE models file not found: {kde_path}"
            )

        if not meta_path.exists():
            raise FileNotFoundError(
                f"[DENSITY FILTER] Metadata file not found: {meta_path}"
            )

        # Load persisted models
        self._umap = joblib.load(umap_path)
        self._kdes = joblib.load(kde_path)
        meta = joblib.load(meta_path)

        # Restore required metadata used by scoring
        self._n_categories = int(meta["n_categories"])
        self._only_on_category = int(meta["only_on_category"])
        self._kde_kernel = meta["kde_kernel"]
        self._kde_bandwidth = float(meta["kde_bandwidth"])

        self._logger.info(
            "[DENSITY FILTER] Loaded models (UMAP=%s, KDEs=%s, meta=%s; "
            "n_categories=%d, only_on=%d).",
            umap_path,
            kde_path,
            meta_path,
            self._n_categories,
            self._only_on_category,
        )

        # Optional: load cached embedding points if present (for debugging/plots)
        train_points_path = self._kde_models_dir / "umap_train_points.npy"
        train_labels_path = self._kde_models_dir / "umap_train_labels.npy"
        val_points_path = self._kde_models_dir / "umap_val_points.npy"
        val_labels_path = self._kde_models_dir / "umap_val_labels.npy"
        
        if train_points_path.exists() and train_labels_path.exists():
            self._Z_train = np.load(train_points_path)
            self._y_train = np.load(train_labels_path)
            self._logger.info(f"Loaded {len(self._Z_train)} training points")
        
        if val_points_path.exists() and val_labels_path.exists():
            self._Z_val = np.load(val_points_path)
            self._y_val = np.load(val_labels_path)
            self._logger.info(f"Loaded {len(self._Z_val)} validation points")

       
