"""Calibrate the GMM n_components of the Frequency Filter, per band.

The filter scores a candidate by the rarity of its central frequency: a
GMM is fit on the frequency distribution and the score is the inverse of
the estimated density. With too many components the GMM overfits the small
SRT sample and the outlier scores become unstable artifacts.

This script fits the GMM over a grid of n_components for each band, then
measures how stable the outlier ranking is between consecutive grid values
(Spearman over the full ranking + overlap of the top-K outliers). It picks
the largest n_components still inside the stable plateau, i.e. the last one
before stability breaks.

It does not modify frequency.py. It only reads results/passed_density.csv
and prints the value to set in the config.

Usage:
    cd /content/davide/srt-anomaly-detection
    python scripts/calibrate_frequency_ncomp.py
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")


# --- parameters ---

# Increasing grid of component counts to test.
N_COMPS = [5, 10, 20, 50, 100]

# Number of bagged GMMs averaged per fit (matches the filter).
N_BAGS = 10

# A step n -> n' is "stable" only if both thresholds hold.
SPEARMAN_MIN = 0.70
OVERLAP_MIN = 0.50
TOP_K = 20

# SRT receiver bands (MHz).
BANDS = {
    "C": (4200.0, 7700.0),
    "K": (18000.0, 26500.0),
}

SEED = 42


# --- helpers ---

def band_of(f_mhz: float) -> str:
    """Map a central frequency to its band label."""
    for name, (lo, hi) in BANDS.items():
        if lo <= f_mhz <= hi:
            return name
    return "OUT"


def raw_scores(X: np.ndarray, n_comp: int, n_bags: int, rng) -> np.ndarray:
    """Anomaly score = 1 / mean ensemble density (rare freq -> high score)."""
    n = len(X)
    pdf_acc = np.zeros(n)
    for _ in range(n_bags):
        idx = rng.integers(0, n, n)  # bootstrap resample
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=int(rng.integers(1_000_000)),
        )
        gmm.fit(X[idx])
        pdf_acc += np.exp(gmm.score_samples(X))
    mean_pdf = pdf_acc / n_bags
    return 1.0 / (mean_pdf + 1e-300)


def rank_vector(order: np.ndarray) -> np.ndarray:
    """Convert a desc-score ordering into a per-sample rank."""
    r = np.empty(len(order))
    r[order] = np.arange(len(order))
    return r


def calibrate_band(freqs_mhz: np.ndarray, n_comps, n_bags, rng):
    """Fit the grid for one band and pick the stable n_components."""
    X = freqs_mhz.reshape(-1, 1).astype(np.float64)
    n = len(X)

    # Skip grid values that exceed the sample size.
    valid = [nc for nc in n_comps if nc <= n]

    rankings, topsets, top5 = {}, {}, {}
    for nc in valid:
        s = raw_scores(X, nc, n_bags, rng)
        order = np.argsort(-s)
        rankings[nc] = order
        topsets[nc] = set(order[:TOP_K].tolist())
        top5[nc] = freqs_mhz[order[:5]]

    # Stability of consecutive grid steps.
    stability = []
    for a, b in zip(valid, valid[1:]):
        rho = spearmanr(rank_vector(rankings[a]), rank_vector(rankings[b])).correlation
        ov = len(topsets[a] & topsets[b]) / float(TOP_K)
        stable = (rho >= SPEARMAN_MIN) and (ov >= OVERLAP_MIN)
        stability.append((a, b, rho, ov, stable))

    # Climb the grid while steps stay stable; stop at the first break.
    selected = valid[0]
    for (a, b, rho, ov, stable) in stability:
        if stable:
            selected = b
        else:
            break

    return selected, stability, top5, valid, n


# --- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/passed_density.csv",
                    help="CSV of candidates that passed the Density Filter.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["band"] = df["frequency_mhz"].apply(band_of)

    rng = np.random.default_rng(SEED)
    result = {}

    for band in ["C", "K"]:
        sub = df[df["band"] == band]
        if sub.empty:
            print(f"\n===== BAND {band}: no candidates, skipped =====")
            continue

        freqs = sub["frequency_mhz"].values
        selected, stability, top5, valid, n = calibrate_band(
            freqs, N_COMPS, N_BAGS, rng
        )
        result[band] = selected

        print(f"\n===== BAND {band}  (n={n} candidates) =====")
        for nc in valid:
            fs = "  ".join(f"{v:.1f}" for v in top5[nc])
            print(f"  n_components={nc:3d} | top-5 freq (MHz): {fs}")
        print("  --- stability (thresholds: Spearman>={:.2f}, overlap>={:.2f}) ---"
              .format(SPEARMAN_MIN, OVERLAP_MIN))
        for (a, b, rho, ov, stable) in stability:
            flag = "stable" if stable else "BREAK"
            print(f"    {a:3d}->{b:3d} | Spearman={rho:.3f} | "
                  f"overlap top-{TOP_K}={ov:.2f} | {flag}")
        print(f"  >>> selected n_components for band {band}: {selected}")

    print("\n" + "=" * 52)
    print("SET THESE IN THE CONFIG (frequency filter):")
    for band, nc in result.items():
        print(f"  n_components_{band}: {nc}")
    print("=" * 52)


if __name__ == "__main__":
    main()