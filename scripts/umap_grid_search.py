#!/usr/bin/env python3
"""
umap_grid_search.py

Grid search over UMAP hyperparameters for the DensityFilter.

Optimized version:
- CC features are extracted ONCE and cached to disk (.npy); subsequent runs
  load the cache instead of re-reading 1.28M tensors.
- UMAP is fitted on a stratified SUBSAMPLE during the search (full-size fits
  were the dominant cost of the original grid: ~2.5 h per combination on
  640k points). Only the winning configuration should then be retrained on
  the full dataset via the standard pipeline (option 2).
- The default grid is reduced, informed by the results of the full grid
  search performed on the Pearson features.

The evaluation metrics are unchanged, so results are directly comparable
with the previous (Pearson) grid search.

Usage
-----
    python scripts/umap_grid_search.py
    python scripts/umap_grid_search.py --subsample 50000 --top-k 3
    python scripts/umap_grid_search.py --recompute-features
    python scripts/umap_grid_search.py --full-grid          # original 36 combos
"""

import argparse
import csv
import itertools
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from umap import UMAP

from src.srtad.config import filters, paths, simulation
from src.srtad.management.cross_correlation_extractor import CrossCorrelationExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("umap_grid_search")

# ── grid definition ───────────────────────────────────────────────────────────

# Reduced grid (default): informed by the full Pearson grid search
N_NEIGHBORS_VALUES = [25, 100]
MIN_DIST_VALUES    = [0.01, 0.1]
METRIC_VALUES      = ["canberra", "cosine", "correlation"]

# Original full grid (use --full-grid)
N_NEIGHBORS_FULL = [15, 25, 50, 100]
MIN_DIST_FULL    = [0.01, 0.1, 0.3]
METRIC_FULL      = ["canberra", "cosine", "correlation"]

# ── fixed hyperparameters (from config) ───────────────────────────────────────

density_cfg     = filters["density"]
KDE_BANDWIDTH   = float(density_cfg["kde_bandwidth"])
KDE_KERNEL      = str(density_cfg["kernel"])
ONLY_ON_CAT     = int(density_cfg["only_on_category"])
ONLY_OFF_CAT    = int(density_cfg["only_off_category"])
N_CATEGORIES    = int(simulation["mask_combinations"])
RANDOM_STATE    = 42

RESULTS_DIR = Path(paths["results"]) / "umap_grid_search"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH    = Path(paths["results"]) / "umap_grid_search.csv"

# Feature cache: tied to the extractor in use. Change the tag when the
# feature definition changes (e.g. "pearson" vs "ccmax") to avoid mixing.
FEATURE_TAG  = "ccmax"
CACHE_DIR    = Path(paths["results"]) / f"cc_features_{FEATURE_TAG}"


# ── feature extraction with cache ─────────────────────────────────────────────

import re as _re
from concurrent.futures import ProcessPoolExecutor

_PATTERN_RX = _re.compile(r"pattern(\d+)\.npy$")

# Module-level worker (must be picklable for ProcessPoolExecutor)
_worker_extractor = None

def _init_worker():
    global _worker_extractor
    _worker_extractor = CrossCorrelationExtractor()

def _process_file(path_str: str):
    """Load one cadence .npy, extract features, return (features, pattern_id)."""
    try:
        m = _PATTERN_RX.search(path_str)
        if not m:
            return None
        cat = int(m.group(1))
        tensor = np.load(path_str).astype(float)
        if tensor.ndim != 3 or tensor.shape[0] != 6:
            return None
        feats = _worker_extractor.extract_features(tensor)
        return feats, cat
    except Exception:
        return None


def load_or_extract_features(recompute: bool = False,
                             n_workers: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cached CC features if available; otherwise extract and cache them.

    Extraction is STREAMING (one tensor in memory per worker at a time, never
    the whole dataset) and PARALLEL across processes.

    Returns
    -------
    X : np.ndarray of shape (N, 15)
    y : np.ndarray of shape (N,) with category labels
    """
    x_path = CACHE_DIR / "X.npy"
    y_path = CACHE_DIR / "y.npy"

    if not recompute and x_path.exists() and y_path.exists():
        log.info("Loading cached features from %s", CACHE_DIR)
        return np.load(x_path), np.load(y_path)

    cadences_dir = Path(paths["data"]) / simulation["output_cadences_dir"]
    files = sorted(str(p) for p in cadences_dir.glob("cadence_*_pattern*.npy"))
    log.info("Found %d cadence files; extracting CC features (tag=%s) "
             "with %d workers...", len(files), FEATURE_TAG, n_workers)

    X, y = [], []
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(files), desc="CC features")
    except ImportError:
        bar = None

    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker) as pool:
        for res in pool.map(_process_file, files, chunksize=256):
            if res is not None:
                X.append(res[0])
                y.append(res[1])
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(x_path, X)
    np.save(y_path, y)
    log.info("Features cached to %s  (X: %s)", CACHE_DIR, X.shape)
    return X, y


def stratified_subsample(X: np.ndarray,
                         y: np.ndarray,
                         n: int,
                         seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified subsample of (X, y) preserving category proportions."""
    if n >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=n,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    return X_sub, y_sub


# ── evaluation metrics (unchanged) ────────────────────────────────────────────

def evaluate(Z_val: np.ndarray,
             y_val: np.ndarray,
             kdes: Dict[int, KernelDensity]) -> Dict[str, float]:
    """
    Compute separability metrics between only-ON and only-OFF clusters.

    Metrics
    -------
    centroid_distance : Euclidean distance between only-ON and only-OFF centroids
    silhouette        : silhouette score restricted to only-ON and only-OFF points
    p_only_on_mean    : mean KDE probability for only-ON validation points
    p_only_on_median  : median KDE probability for only-ON validation points
    """
    mask_on  = y_val == ONLY_ON_CAT
    mask_off = y_val == ONLY_OFF_CAT

    Z_on  = Z_val[mask_on]
    Z_off = Z_val[mask_off]

    # Centroid distance
    if len(Z_on) > 0 and len(Z_off) > 0:
        centroid_on  = Z_on.mean(axis=0)
        centroid_off = Z_off.mean(axis=0)
        centroid_dist = float(np.linalg.norm(centroid_on - centroid_off))
    else:
        centroid_dist = 0.0

    # Silhouette score (subset of only-ON and only-OFF)
    n_sil = min(2000, len(Z_on), len(Z_off))
    if n_sil >= 2:
        rng = np.random.default_rng(RANDOM_STATE)
        idx_on  = rng.choice(len(Z_on),  n_sil, replace=False)
        idx_off = rng.choice(len(Z_off), n_sil, replace=False)
        Z_sil = np.vstack([Z_on[idx_on], Z_off[idx_off]])
        y_sil = np.array([0] * n_sil + [1] * n_sil)
        try:
            sil = float(silhouette_score(Z_sil, y_sil))
        except Exception:
            sil = 0.0
    else:
        sil = 0.0

    # KDE probability for only-ON validation points
    kde_on = kdes.get(ONLY_ON_CAT)
    if kde_on is not None and len(Z_on) > 0:
        log_probs = kde_on.score_samples(Z_on)
        probs     = np.exp(log_probs)
        p_mean    = float(probs.mean())
        p_median  = float(np.median(probs))
    else:
        p_mean = p_median = 0.0

    return {
        "centroid_distance": centroid_dist,
        "silhouette":        sil,
        "p_only_on_mean":    p_mean,
        "p_only_on_median":  p_median,
    }


# ── UMAP plot (unchanged) ─────────────────────────────────────────────────────

def plot_umap(Z_val: np.ndarray,
              y_val: np.ndarray,
              n_neighbors: int,
              min_dist: float,
              metric: str,
              metrics: Dict[str, float],
              rank: int) -> None:
    """Save a UMAP scatter plot for a given configuration."""
    fig, ax = plt.subplots(figsize=(8, 6))

    mask_on  = y_val == ONLY_ON_CAT
    mask_off = y_val == ONLY_OFF_CAT
    mask_mix = ~mask_on & ~mask_off

    # Subsample background for speed
    n_bg = min(10_000, mask_mix.sum())
    rng  = np.random.default_rng(RANDOM_STATE)
    idx  = rng.choice(np.where(mask_mix)[0], n_bg, replace=False)

    ax.scatter(Z_val[idx, 0], Z_val[idx, 1],
               s=1, c="#888888", alpha=0.05, label="Mixed/noise")
    ax.scatter(Z_val[mask_off, 0], Z_val[mask_off, 1],
               s=4, c="#ff5064", alpha=0.3, label="Only-OFF")
    ax.scatter(Z_val[mask_on, 0], Z_val[mask_on, 1],
               s=4, c="#64ff50", alpha=0.3, label="Only-ON")

    title = (f"Rank #{rank} | n_neighbors={n_neighbors}, "
             f"min_dist={min_dist}, metric={metric}\n"
             f"centroid_dist={metrics['centroid_distance']:.3f}, "
             f"silhouette={metrics['silhouette']:.3f}, "
             f"p_ON_mean={metrics['p_only_on_mean']:.4f}")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("UMAP X")
    ax.set_ylabel("UMAP Y")
    ax.legend(markerscale=4, fontsize=8)
    plt.tight_layout()

    fname = (f"rank{rank:02d}_nn{n_neighbors}_"
             f"md{str(min_dist).replace('.','')}_"
             f"{metric}.png")
    fig.savefig(RESULTS_DIR / fname, dpi=100)
    plt.close(fig)
    log.info("Saved plot: %s", fname)


# ── single run (unchanged logic) ──────────────────────────────────────────────

def run_single(X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: np.ndarray,
               y_val: np.ndarray,
               n_neighbors: int,
               min_dist: float,
               metric: str) -> Dict[str, Any]:
    """
    Train UMAP + KDEs for one hyperparameter combination and return metrics.
    """
    t0 = time.time()

    umap = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    Z_train = umap.fit_transform(X_train)
    Z_val_  = umap.transform(X_val)

    # Fit KDEs
    kdes: Dict[int, KernelDensity] = {}
    for k in range(N_CATEGORIES):
        mask = y_train == k
        if not mask.any():
            continue
        kde = KernelDensity(kernel=KDE_KERNEL, bandwidth=KDE_BANDWIDTH)
        kde.fit(Z_train[mask])
        kdes[k] = kde

    metrics = evaluate(Z_val_, y_val, kdes)
    elapsed = time.time() - t0

    return {
        "n_neighbors":      n_neighbors,
        "min_dist":         min_dist,
        "metric":           metric,
        "elapsed_s":        round(elapsed, 1),
        "Z_val":            Z_val_,
        "y_val":            y_val,
        **metrics,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grid search over UMAP hyperparameters for DensityFilter"
    )
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top configurations to plot (default: 5)")
    parser.add_argument("--subsample", type=int, default=64_000,
                        help="Per-split stratified subsample size used for the "
                             "search (default: 64000). Use 0 to disable.")
    parser.add_argument("--recompute-features", action="store_true",
                        help="Ignore the feature cache and re-extract")
    parser.add_argument("--full-grid", action="store_true",
                        help="Use the original 36-combination grid")
    args = parser.parse_args()

    # Features (cached)
    X, y = load_or_extract_features(recompute=args.recompute_features)
    log.info("Features shape: %s", X.shape)

    # Train/val split (same as DensityFilter.fit)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y,
    )

    # Stratified subsampling for the search
    if args.subsample and args.subsample > 0:
        X_train, y_train = stratified_subsample(X_train, y_train, args.subsample, RANDOM_STATE)
        X_val,   y_val   = stratified_subsample(X_val,   y_val,   args.subsample, RANDOM_STATE + 1)
        log.info("Subsampled -> Train: %d  Val: %d", len(X_train), len(X_val))
    else:
        log.info("Train: %d  Val: %d (no subsampling)", len(X_train), len(X_val))

    # Grid
    if args.full_grid:
        combos = list(itertools.product(N_NEIGHBORS_FULL, MIN_DIST_FULL, METRIC_FULL))
    else:
        combos = list(itertools.product(N_NEIGHBORS_VALUES, MIN_DIST_VALUES, METRIC_VALUES))
    log.info("Running %d combinations...", len(combos))

    results: List[Dict[str, Any]] = []

    for i, (nn, md, metric) in enumerate(combos, 1):
        log.info("[%d/%d] n_neighbors=%d, min_dist=%s, metric=%s",
                 i, len(combos), nn, md, metric)
        try:
            r = run_single(X_train, y_train, X_val, y_val, nn, md, metric)
            results.append(r)
            log.info("  centroid_dist=%.3f  silhouette=%.3f  p_ON_mean=%.4f  (%.1fs)",
                     r["centroid_distance"], r["silhouette"],
                     r["p_only_on_mean"], r["elapsed_s"])
        except Exception as e:
            log.warning("  FAILED: %s", e)
            continue

    # Sort by centroid_distance (primary) + silhouette (secondary)
    results.sort(key=lambda r: (r["centroid_distance"] + r["silhouette"]), reverse=True)

    # Save CSV
    fieldnames = ["n_neighbors", "min_dist", "metric", "centroid_distance",
                  "silhouette", "p_only_on_mean", "p_only_on_median", "elapsed_s"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    log.info("Results saved to: %s", CSV_PATH)

    # Print top results
    print(f"\n{'Rank':>4}  {'n_neighbors':>12}  {'min_dist':>8}  "
          f"{'metric':>12}  {'centroid':>10}  {'silhouette':>10}  {'p_ON_mean':>10}")
    print("-" * 80)
    for rank, r in enumerate(results[:10], 1):
        print(f"{rank:>4}  {r['n_neighbors']:>12}  {r['min_dist']:>8}  "
              f"{r['metric']:>12}  {r['centroid_distance']:>10.3f}  "
              f"{r['silhouette']:>10.3f}  {r['p_only_on_mean']:>10.4f}")

    # Plot top-k
    log.info("Generating plots for top %d configurations...", args.top_k)
    for rank, r in enumerate(results[:args.top_k], 1):
        plot_umap(r["Z_val"], r["y_val"],
                  r["n_neighbors"], r["min_dist"], r["metric"],
                  {k: r[k] for k in ["centroid_distance", "silhouette", "p_only_on_mean"]},
                  rank)

    print(f"\nBest configuration:")
    best = results[0]
    print(f"  n_neighbors = {best['n_neighbors']}")
    print(f"  min_dist    = {best['min_dist']}")
    print(f"  metric      = {best['metric']}")
    print(f"  Update config/default.yaml with these values and retrain the density model")
    print(f"  (option 2) on the FULL dataset.")


if __name__ == "__main__":
    main()