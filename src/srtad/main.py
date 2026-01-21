import sys
from pathlib import Path
from typing import List, Tuple
from src.srtad.simulation.generator import SimulationGenerator
from src.srtad.core.dataset import Dataset
from src.srtad.ml.filters.density import DensityFilter
from src.srtad.config import paths, simulation as sim_cfg, filters
from src.srtad.core.candidate import Candidate
from scripts.category import create_category_report
from joblib import Parallel, delayed
from src.srtad.ml.filters.frequency import FrequencyFilter
from src.srtad.ml.filters.similarity import SimilarityFilter

# Per-process cache to avoid re-instantiating filters for every candidate
_FREQ = None
_SIM = None

def run_fit_density() -> None:
    """
    Train the DensityFilter (UMAP + KDE) using simulated cadences.

    This step is executed once and persists the trained density model to disk.
    The resulting model is later used during inference on real candidates.
    """
    ds = Dataset()

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

def run_density_filter() -> List[Candidate]:
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
    ds = Dataset()
    passed_candidates : List[Candidate] = []
    density = DensityFilter()
    threshold = filters["density"]["threshold"]

    candidates = ds.load(real_dir)

    try:
        for candidate in candidates:
            score = density.calculate(candidate)
            candidate.set_density_score(score)

            if(candidate.density_score >= threshold):
                passed_candidates.append(candidate)
    
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("You must train the Density model first (option 2).")

    print(f"Filtered {len(passed_candidates)} candidates on {len(candidates)}")

    print("Generating category PDF for ALL candidates.")
    # The category report is generated on the full set to allow manual inspection 
    # even when no candidate passes the density threshold
    create_category_report(candidates)
    
    return passed_candidates

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

def run_frequency_similarity_filters(candidates: List[Candidate]) -> None:
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
    if not candidates:
        print("No candidates passed the density filter. Skipping.")
        return

    freq = FrequencyFilter()
    sim = SimilarityFilter()
    freq.fit(candidates)
    sim.fit(candidates)

    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_score_one)(i, c) for i, c in enumerate(candidates)
    )

    for idx, f_score, s_score in results:
        candidates[idx].set_frequency_score(f_score)
        candidates[idx].set_similarity_score(s_score)

    print(f"Computed frequency+similarity scores for {len(candidates)} candidates.")

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
    passed_candidates: List[Candidate] = []

    while True:
        print("\n=== SRT Anomaly Detection ===")
        print("1) Generate Synthetic Cadences")
        print("2) Train Density Model (simulated data)")
        print("3) Run Density Filter")
        print("4) Run Frequency + Similarity Filters")
        print("0) Exit")

        choice = input("Select: ").strip()

        if choice == "1":
            SimulationGenerator().run()
        elif choice == "2":
            run_fit_density()
        elif choice =="3":
            passed_candidates = run_density_filter()
        elif choice == "4":
            run_frequency_similarity_filters(passed_candidates)
        elif choice == "0":
          sys.exit(0)
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
