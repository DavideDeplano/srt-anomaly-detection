from pathlib import Path
from typing import List, Tuple
from joblib import Parallel, delayed
import sys
import csv

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

def run_density_filter() -> Tuple[List[Candidate], List[Candidate]]:
    """
    Apply the DensityFilter to real candidates.

    Workflow:
    - Load all real candidates from paths["real_png_dir"]
    - Compute a density score for each candidate
    - Retain only candidates whose score exceeds the configured threshold

    Side effect:
    - A category report PDF is generated for ALL candidates
      (both passed and rejected) to allow manual inspection.
    """
    real_dir = Path(paths["real_png_dir"])
    ds = Dataset(png_dir=real_dir, use_tqdm=True)
    passed_candidates : List[Candidate] = []
    density = DensityFilter()
    threshold = filters["density"]["threshold"]

    candidates = ds.load()

    try:
      it = candidates
      if tqdm is not None:
          it = tqdm(candidates, desc="Density scoring", unit="candidate")

      for candidate in it:
          score = density.calculate(candidate)
          candidate.set_density_score(score)

          if candidate.density_score >= threshold:
              passed_candidates.append(candidate)

    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("You must train the Density model first (option 2).")
        return [], []

    print(f"Filtered {len(passed_candidates)} candidates on {len(candidates)}")

    out_csv = Path(paths["results"]) / "passed_density.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

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

    print("Generating category PDF for ALL candidates.")
    # The category report is generated on the full set to allow manual inspection 
    # even when no candidate passes the density threshold
    create_category_report(candidates)
    
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
    passed_candidates: List[Candidate],
    all_candidates: List[Candidate],
) -> None:
    """
    Fit and apply FrequencyFilter and SimilarityFilter on density-passed candidates.

    Pipeline:
    1) Fit both filters on the provided candidate list and persist models to disk.
    2) Compute frequency and similarity scores for each candidate in parallel
       using CPU-based multiprocessing.

    Notes:
    - Training is executed once in the main process.
    - Scoring is parallelized across candidates.
    """
    if not passed_candidates:
        print("No candidates passed the density filter. Skipping.")
        return
    if not all_candidates:
        print("No all_candidates available. Run density filter first (option 3).")
        return

    freq = FrequencyFilter()
    sim = SimilarityFilter()

    freq.fit(passed_candidates)
    sim.fit(passed_candidates)

    for c in passed_candidates:
        b = FrequencyFilter._extract_band(c)
        if b is not None:
            c.set_band(b)

    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_score_one)(i, c) for i, c in enumerate(passed_candidates)
    )

    for idx, f_score, s_score in results:
        passed_candidates[idx].set_frequency_score(f_score)
        passed_candidates[idx].set_similarity_score(s_score)

    viz = Visualizer()
    viz.plot_frequency_histogram_by_band(all_candidates, filename="frequency_hist.png")
    viz.plot_frequency_score_histogram_by_band(passed_candidates, filename="freqscore_hist.png")

    sim.plot_umap_similarity(
        background_candidates=all_candidates,
        scored_candidates=passed_candidates,
        filename="similarity_umap.png",
    )

    print(f"Computed frequency+similarity scores for {len(passed_candidates)} candidates.")

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
        print("No candidates available for ranking. Run filters first.")
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

def main() -> None:
    """
    Command-line interface for the SRT anomaly detection pipeline.

    Menu options:
    1) Generate synthetic cadences
    2) Train density model on simulated data
    3) Run density filter on real candidates
    4) Fit and apply frequency + similarity filters
    0) Exit
    """
    passed_candidates = []
    all_candidates = []

    while True:
        print("\n=== SRT Anomaly Detection ===")
        print("1) Generate Synthetic Cadences")
        print("2) Train Density Model (simulated data)")
        print("3) Run Density Filter")
        print("4) Run Frequency + Similarity Filters")
        print("5) Run Ranking")
        print("0) Exit")

        choice = input("Select: ").strip()

        if choice == "1":
            SimulationGenerator().run()
        elif choice == "2":
            run_fit_density()
        elif choice =="3":
            passed_candidates, all_candidates = run_density_filter()
        elif choice == "4":
            run_frequency_similarity_filters(passed_candidates, all_candidates)
        elif choice == "5":
            run_ranking(passed_candidates)
        elif choice == "0":
          sys.exit(0)
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
