# SRT Anomaly Detection

This repository contains a modular anomaly-detection pipeline for
SETI-like radio signals, combining simulated cadences and real data
from the Sardinia Radio Telescope (SRT).

The entire pipeline is an implementation and adaptation of the
methodology described in *Pardo et al.*,
available in [`Pardo_2025_AJ_170_12.pdf`](docs/paper/Pardo_2025_AJ_170_12.pdf),
reworked to operate on SRT data and structured as a reproducible,
modular software framework.

Candidates are first filtered using a density-based model trained on
simulated cadences, where each 6-panel observation is represented by
cross-correlation (CC) features and embedded via UMAP with per-category
KDEs. The remaining candidates are then independently scored using a
frequency-based Gaussian Mixture Model (GMM) ensemble and an ON/OFF
similarity metric in a UMAP-embedded panel space, and finally ranked to
prioritize human inspection.

---

## Repository structure
- config/
  - default.yaml      Global configuration file
- docs/               Documentation
- scripts/
  - unix/             Linux/macOS setup and run scripts
  - win/              Windows setup and run scripts
  - category.py       Category report generation
  - noise_extractor.py  Background noise extraction
- src/srtad/
  - core/
    - candidate.py    Candidate representation
    - dataset.py      Dataset loading and handling
  - management/
    - cross_correlation_extractor.py  CC feature extraction
    - ranker.py       Ranking logic
    - visualizer.py   Plotting and reports
  - ml/filters/
    - density.py      Density-based filter (UMAP + KDE)
    - frequency.py    Frequency-based scoring (GMM)
    - similarity.py   ON/OFF similarity scoring
    - i_filter.py     Filter interface
  - simulation/
    - generator.py    Synthetic cadence generation
  - utils/
    - logger.py       Logging utilities
  - config.py         Configuration loader
  - main.py           Pipeline entry point
- env.yml             Conda environment specification

Note: directories such as `data/`, `models/`, `results/`, and `logs/` are created at runtime
by the pipeline scripts and are not versioned in the repository.

---

## Environment Setup 

The project uses **Conda** for environment management.
To automatically create and configure the `srt-anom` environment:

### Windows systems
Run:
```bash
scripts\win\setup_env.bat
```
This script:

- creates or updates the `srt-anom` Conda environment from `env.yml`;

- installs all required dependencies;

- registers the project for direct execution.
  
### Running the project

After the initial setup, the pipeline can be executed with:

```bash
scripts\win\run.bat
```

There is no need to manually activate the Conda environment:
`run.bat` automatically runs the project in the correct environment.

### Linux / macOS systems

Use the equivalent scripts located in the `scripts/unix` directory:

```bash
scripts/unix/setup_env.sh
scripts/unix/run.sh
```

---

## Generated outputs

The pipeline produces the following artifacts during execution:

- `passed_density.csv`  
  Candidates passing the density-based pre-filter.

- `top_k_combined.csv`  
  Final ranked candidates using combined frequency and similarity scores.

- `random_control.csv`  
  Random control sample used for statistical comparison.

- `REPORT_CATEGORIES.pdf`  
  Diagnostic category report for manual inspection.

All outputs are written under the runtime `results/` directory.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
