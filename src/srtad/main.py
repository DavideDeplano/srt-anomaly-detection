from pathlib import Path
from typing import List, Tuple
from joblib import Parallel, delayed
from collections import Counter
import numpy as np
import sys
import csv
import pickle

from src.srtad.simulation.generator import SimulationGenerator
from src.srtad.core.dataset import Dataset
from src.srtad.ml.filters.density import DensityFilter
from src.srtad.config import paths, simulation as sim_cfg, filters
from src.srtad.core.candidate import Candidate
from scripts.category import create_category_report
from src.srtad.ml.filters.frequency import FrequencyFilter
from src.srtad.ml.filters.similarity import SimilarityFilter
from src.srtad.management.ranker import Ranker
from src.srtad.management.visualizer import Visualizer
from scripts.umap_reverse_search import UMAPReverseSearch
from src.srtad.ml.filters.autoencoder import AutoencoderFilter

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# Per-process cache to avoid re-instantiating filters for every candidate
_FREQ = None
_SIM = None

def run_fit_density() -> None:
    """
    Train the DensityFilter (UMAP + KDE) using simulated cadences.

    This step is executed once and persists the trained density model to disk.
    The resulting model is later used during inference on real candidates.
    """
    ds = Dataset(use_tqdm=True)

    base_data_dir = Path(paths["data"])
    cadences_dir = base_data_dir / sim_cfg["output_cadences_dir"]

    print(f"\nLoading simulated cadences from: {cadences_dir}")

    try:
        simulated = ds.load_simulated_cadences(cadences_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run option 1 (Generate synthetic cadences) first.")
        return

    print(f"Loaded {len(simulated)} simulated cadences.")
    print("Fitting DensityFilter (UMAP + KDE) on simulated data...")

    density = DensityFilter()
    density.fit(simulated)

    print("Density model training completed and saved.\n")

def run_density_filter() -> tuple:
    """
    Apply DensityFilter to real candidates with adaptive threshold
    and Figure 4-style diagnostic plot.
    """
    real_dir = Path(paths["real_png_dir"])
    ds = Dataset(png_dir=real_dir, use_tqdm=True)
    density = DensityFilter()
    
    candidates = ds.load()
    
    # Collect scores for ALL candidates
    all_p_only_on = []      # P_only_on for every candidate (for Figure 4 plot)
    passed_p_only_on = []   # P_only_on only where argmax == only_on
    
    try:
        it = candidates
        if tqdm is not None:
            it = tqdm(candidates, desc="Density scoring", unit="candidate")

        for candidate in it:
            score, p_on, best_cat = density.calculate_with_details(candidate)
            candidate.set_density_score(score)
            
            # Collect P_only_on for ALL candidates (for the histogram)
            all_p_only_on.append(p_on)
            
            # Collect only-on scores where argmax matched
            if score > 0:
                passed_p_only_on.append(score)

    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return [], []

    all_p_only_on = np.array(all_p_only_on, dtype=float)
    passed_p_only_on = np.array(passed_p_only_on, dtype=float)
    
    # Save arrays for later analysis
    results_dir = Path(paths["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "density_all_p_only_on.npy", all_p_only_on)
    np.save(results_dir / "density_passed_p_only_on.npy", passed_p_only_on)
    
    # Plot and compute adaptive threshold
    viz = Visualizer()
    threshold = viz.plot_density_histogram(
        all_probs=all_p_only_on,
        passed_probs=passed_p_only_on,
        threshold=None,  # let it compute adaptively
        filename="density_real_hist.png"
    )
    
    print(f"[DENSITY] Adaptive threshold: {threshold:.6f}")
    print(f"[DENSITY] Candidates with argmax=Cat42: {len(passed_p_only_on)}")
    print(f"[DENSITY] Candidates with P_only_on > 0: {np.sum(all_p_only_on > 0)}")
    
    # Apply threshold
    passed_candidates = [
        c for c in candidates
        if c.density_score >= threshold
    ]
    
    print(f"[DENSITY] Passed threshold: {len(passed_candidates)} / {len(candidates)}")

    # CSV output
    out_csv = results_dir / "passed_density.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "density_score", "frequency_mhz", "source_path"])
        for c in passed_candidates:
            writer.writerow([
                c.id,
                f"{c.density_score:.6e}",
                f"{c.frequency_hz / 1e6:.6f}",
                str(c.source_path),
            ])
    print(f"[OK] CSV written: {out_csv}")

    # Category report
    print("Generating category PDF for ALL candidates.")
    create_category_report(candidates)

    # Save pickle
    out_pkl = results_dir / "passed_candidates.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump((passed_candidates, candidates), f)
    print(f"[OK] Saved candidates to: {out_pkl}")
    
    return passed_candidates, candidates

def _get_filters() -> Tuple[FrequencyFilter, SimilarityFilter]:
    """
    Lazily instantiate FrequencyFilter and SimilarityFilter once per worker process.

    Each worker process maintains its own cache. Model parameters are loaded
    automatically from disk on first use via calculate().
    """
    global _FREQ, _SIM
    if _FREQ is None:
        _FREQ = FrequencyFilter()   
    if _SIM is None:
        _SIM = SimilarityFilter()
    return _FREQ, _SIM

def _score_one(idx: int, c: Candidate) -> Tuple[int, float, float]:
    """
    Compute frequency and similarity scores for a single candidate.

    Returns:
        Tuple containing:
        - index of the candidate in the original list
        - frequency score
        - similarity score
    """
    freq, sim = _get_filters()
    return idx, float(freq.calculate(c)), float(sim.calculate(c))

def run_frequency_similarity_filters(
    passed_candidates,
    all_candidates,
):
    """
    Fit and apply FrequencyFilter and SimilarityFilter on density-passed candidates.
    
    Adaptive behavior:
    - Frequency bin cut is skipped when candidate count is below 200
      (paper designed for ~10^6 candidates; aggressive binning on small
      samples removes too many candidates)
    - Band labels are assigned to ALL candidates for diagnostic plots
    """
    if not passed_candidates or not all_candidates:
        pkl_path = Path(paths["results"]) / "passed_candidates.pkl"
        if pkl_path.exists():
            import pickle
            with open(pkl_path, "rb") as f:
                passed_candidates, all_candidates = pickle.load(f)
            print(f"[OK] Loaded {len(passed_candidates)} passed and "
                f"{len(all_candidates)} total candidates from {pkl_path}")
        else:
            print("No candidates available. Run density filter first (option 3).")
            return []

    # ---- ADAPTIVE FREQUENCY CUT ----
    # The paper's 5% noisy-bin cut was designed for ~10^6 candidates.
    # With < 200 candidates it removes too many; skip it.
    MIN_CANDIDATES_FOR_FREQ_CUT = 200
    
    if len(passed_candidates) >= MIN_CANDIDATES_FOR_FREQ_CUT:
        freq_hz = np.array([c.frequency_hz for c in passed_candidates])
        n_bins = min(200, max(10, len(passed_candidates) // 10))
        bin_edges = np.linspace(freq_hz.min(), freq_hz.max(), n_bins + 1)
        bin_indices = np.digitize(freq_hz, bin_edges, right=False)

        bin_counts = Counter(bin_indices.tolist())
        counts_array = np.array(list(bin_counts.values()))
        threshold_count = np.percentile(counts_array, 95)
        noisy_bins = {b for b, n in bin_counts.items() if n > threshold_count}

        n_before = len(passed_candidates)
        passed_candidates = [
            c for i, c in enumerate(passed_candidates)
            if bin_indices[i] not in noisy_bins
        ]
        n_after = len(passed_candidates)

        print(f"[FREQUENCY CUT] Removed {n_before - n_after} candidates "
            f"({100*(n_before-n_after)/max(n_before,1):.1f}%) "
            f"from {len(noisy_bins)} noisy frequency bins (top 5%).")
    else:
        print(f"[FREQUENCY CUT] Skipped — only {len(passed_candidates)} candidates "
              f"(need >= {MIN_CANDIDATES_FOR_FREQ_CUT} for frequency bin cut to be meaningful).")

    freq = FrequencyFilter()
    sim = SimilarityFilter()

    freq.fit(passed_candidates)
    sim.fit(passed_candidates)

    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_score_one)(i, c) for i, c in enumerate(passed_candidates)
    )

    for idx, f_score, s_score in results:
        passed_candidates[idx].set_frequency_score(f_score)
        passed_candidates[idx].set_similarity_score(s_score)

    # ---- SET BAND FOR ALL CANDIDATES (needed for diagnostic plots) ----
    for c in all_candidates:
        if getattr(c, "band", None) is None:
            c.set_band(FrequencyFilter.extract_band(c))

    viz = Visualizer()
    viz.plot_frequency_histogram_by_band(all_candidates, filename="frequency_hist.png")
    viz.plot_frequency_score_histogram_by_band(passed_candidates, filename="freqscore_hist.png")

    sim.plot_umap_similarity(
        background_candidates=all_candidates,
        scored_candidates=passed_candidates,
        filename="similarity_umap.png",
    )

    print(f"Computed frequency+similarity scores for {len(passed_candidates)} candidates.")
    return passed_candidates

def run_ranking(candidates: List[Candidate]) -> None:
    """
    Run paper-style ranking samples on candidates already scored by
    FrequencyFilter and SimilarityFilter.

    Requirements:
    - candidate.frequency_score and candidate.similarity_score must be set
    - candidate.band must be set (e.g. by FrequencyFilter.calculate)
    - candidate.target must be set (e.g. Dataset.load -> candidate.set_target("UNKNOWN"))
    """
    if not candidates:
        pkl_path = Path(paths["results"]) / "passed_candidates.pkl"
        if pkl_path.exists():
            import pickle
            with open(pkl_path, "rb") as f:
                candidates, _ = pickle.load(f)
            print(f"[OK] Loaded {len(candidates)} candidates from {pkl_path}")
        else:
            print("No candidates available. Run filters first.")
            return
    
    # Exclude bands not modeled 
    candidates = [c for c in candidates if c.band != "OUT_OF_BAND"]

    r = Ranker()
    samples = r.build_samples(candidates)

    # Print summary
    print("\n=== Ranking samples ===")
    for name, group in samples.items():
        print(f"{name}: {len(group)}")

    # Write CSVs under results/ (no extra Ranker methods needed)
    out_dir = Path(paths["results"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_csv(name: str, group: List[Candidate]) -> None:
        out_csv = out_dir / f"{name}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "id",
                "target",
                "band",
                "density_score",
                "frequency_score",
                "similarity_score",
                "frequency_mhz",
                "drift_hz_s",
                "source_path",
            ])
            for c in group:
                w.writerow([
                    c.id,
                    getattr(c, "target", ""),
                    getattr(c, "band", ""),
                    "" if c.density_score is None else f"{float(c.density_score):.6e}",
                    "" if c.frequency_score is None else f"{float(c.frequency_score):.6f}",
                    "" if c.similarity_score is None else f"{float(c.similarity_score):.6f}",
                    f"{float(c.frequency_hz) / 1e6:.6f}",
                    f"{float(c.drift_hz_s):.6f}",
                    str(c.source_path),
                ])
        print(f"[OK] CSV written: {out_csv}")

    for name, group in samples.items():
        _write_csv(name, group)

def run_fit_autoencoder() -> None:
    """Train the AutoencoderFilter (Hyperband tuning) on simulated cadences."""
    from pathlib import Path
    from src.srtad.ml.filters.autoencoder import _TUNER_DIR, _X_CACHE_PATH

    ae = AutoencoderFilter()

    # Se X_cache esiste, passa un iterabile vuoto — fit() lo ignorerà
    # e caricherà X direttamente dalla cache
    if _X_CACHE_PATH.exists():
        print(f"\n[AUTOENCODER] X cache found — skipping data loading.")
        ae.fit(iter([]))  # iterabile vuoto, fit() usa la cache
        print("[OK] Autoencoder training complete.")
        return

    # Altrimenti carica i dati normalmente
    ds = Dataset(use_tqdm=True)
    base_data_dir = Path(paths["data"])
    cadences_dir = base_data_dir / sim_cfg["output_cadences_dir"]

    print(f"\nLoading simulated cadences from: {cadences_dir}")
    try:
        simulated = ds.load_simulated_cadences(cadences_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run option 1 (Generate synthetic cadences) first.")
        return

    print(f"Loaded {len(simulated)} simulated cadences.")
    ae.fit(simulated)
    print("[OK] Autoencoder training complete.")

def run_autoencoder_filter() -> List[Candidate]:
    """Score real candidates with the AutoencoderFilter and save results."""
    real_dir = Path(paths["real_png_dir"])
    ds = Dataset(png_dir=real_dir, use_tqdm=True)
    candidates = ds.load()

    if not candidates:
        print("No real candidates found. Check real_png_dir in config.")
        return []

    ae = AutoencoderFilter()
    ae.calibrate(candidates)

    it = candidates
    if tqdm is not None:
        it = tqdm(candidates, desc="Autoencoder scoring", unit="candidate")

    for c in it:
        c.set_ae_score(ae.calculate(c))

    results_dir = Path(paths["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    out_pkl = results_dir / "passed_candidates.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump((candidates, candidates), f)

    out_csv = results_dir / "passed_autoencoder.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "ae_score", "frequency_mhz", "drift_hz_s", "source_path"])
        for c in candidates:
            w.writerow([
                c.id,
                f"{float(c.ae_score):.6f}" if c.ae_score is not None else "",
                f"{float(c.frequency_hz) / 1e6:.6f}",
                f"{float(c.drift_hz_s):.6f}",
                str(c.source_path),
            ])

    print(f"[OK] {len(candidates)} candidates scored. Results saved to {out_csv}")
    return candidates

def main() -> None:
    """
    Command-line interface for the SRT anomaly detection pipeline.

    Workflow — Pardo (DensityFilter):    1 → 2 → 3 → 4 → 5
    Workflow — Autoencoder extension:    1 → 7 → 8 → 4 → 5
    """
    passed_candidates = []
    all_candidates = []

    while True:
        print("\n=== SRT Anomaly Detection ===")
        print("1) Generate Synthetic Cadences")
        print("--- Pardo pipeline ---")
        print("2) Train Density Model (simulated data)")
        print("3) Run Density Filter")
        print("--- Autoencoder extension ---")
        print("7) Train Autoencoder (simulated data)")
        print("8) Run Autoencoder Filter")
        print("--- Common steps ---")
        print("4) Run Frequency + Similarity Filters")
        print("5) Run Ranking")
        print("6) UMAP Reverse Search")
        print("0) Exit")

        choice = input("Select: ").strip()

        if choice == "1":
            SimulationGenerator().run()
        elif choice == "2":
            run_fit_density()
        elif choice =="3":
            passed_candidates, all_candidates = run_density_filter()
        elif choice == "4":
            passed_candidates = run_frequency_similarity_filters(passed_candidates, all_candidates)
        elif choice == "5":
            run_ranking(passed_candidates)
        elif choice == "6":
            searcher = UMAPReverseSearch()
            searcher.load()
            fig = searcher.plot()
        elif choice == "7":
            run_fit_autoencoder()
        elif choice == "8":
            passed_candidates = run_autoencoder_filter()
            all_candidates = passed_candidates
        elif choice == "0":
          sys.exit(0)
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
