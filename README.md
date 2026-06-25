# SRT Anomaly Detection

A modular anomaly-detection pipeline for SETI-like radio signals, combining simulated cadences and real observations from the **Sardinia Radio Telescope (SRT)** in C-band (4.2–7.7 GHz) and K-band (18–26.5 GHz).

The pipeline is an independent implementation and adaptation of the methodology of *Pardo et al.* (2025), reworked for SRT data and extended with a convolutional-autoencoder filter as an alternative first-stage screening.

The reference paper is included in [`docs/paper/Pardo_2025_AJ_170_12.pdf`](docs/paper/Pardo_2025_AJ_170_12.pdf).

---

## Overview

Each input cadence is a 6-panel ON/OFF observation. Candidates flow through a sequence of independent scoring filters and a final ranking module:

1. **First-stage screening** (one of two alternatives)
   - **Density Filter** — 15-dimensional cross-correlation features extracted from panel pairs, projected via UMAP, per-category KDE scoring with adaptive threshold.
   - **Autoencoder Filter** — convolutional autoencoder trained on simulated cadences (excluding the only-on-target category); anomaly score from the normalized reconstruction error, calibrated on the real candidate pool.
2. **Frequency Filter** — GMM ensemble (with bagging) on center-frequency distributions, fitted per band.
3. **Similarity Filter** — UMAP projection of per-panel min-max normalized cadences, ON/OFF coherence score.
4. **Ranking** — four selection samples: top frequency percentile, top similarity percentile, top-K combined, random control.

### Adaptations to SRT data

- **NCC-max extractor** — generalization of the cross-correlation features to be tolerant of small frequency-drift translations between panels.
- **Adaptive Density threshold** — first local minimum of the smoothed P(only-on) histogram, with 25th-percentile fallback when the distribution is unimodal.
- **Synthetic-data noise** — background built from multiple SRT sessions to capture instrumental variability.
- **GMM components selected per-band** — via Spearman / top-overlap stability analysis to avoid overfitting on smaller post-Density pools.

---

## Repository structure

```
config/
  default.yaml                       Global configuration
docs/
  paper/                             Reference paper PDF
scripts/
  unix/                              Linux/macOS setup and run
  win/                               Windows setup and run
  category.py                        Category report generation
  noise_extractor.py                 Background-noise extraction from SRT data
src/srtad/
  core/
    candidate.py                     Candidate representation (preprocessed + cadence_raw)
    dataset.py                       Dataset loading and preprocessing
  management/
    cross_correlation_extractor.py   CC / NCC-max feature extraction
    ranker.py                        Ranking logic (4 samples)
    visualizer.py                    Plots and report generation
  ml/filters/
    i_filter.py                      Filter interface
    density.py                       Density Filter (UMAP + KDE)
    frequency.py                     Frequency Filter (GMM ensemble)
    similarity.py                    Similarity Filter (UMAP)
    autoencoder.py                   Autoencoder Filter (convolutional AE)
  simulation/
    generator.py                     Synthetic cadence generation (Setigen)
  utils/
    logger.py                        Logging utilities
  config.py                          Configuration loader
  main.py                            Pipeline entry point
env.yml                              Conda environment specification
```

Note: runtime directories (`data/`, `models/`, `results/`, `logs/`) are created on first run and are not versioned.

---

## Environment setup

The project uses **Conda** for environment management. The setup scripts create or update the `srt-anom` environment from `env.yml` and register the project for direct execution — no manual `conda activate` is needed.

### Windows

```bat
scripts\win\setup_env.bat
```

### Linux / macOS

```bash
scripts/unix/setup_env.sh
```

---

## Running the pipeline

After setup, run:

### Windows

```bat
scripts\win\run.bat
```

### Linux / macOS

```bash
scripts/unix/run.sh
```

`run.bat` / `run.sh` execute the project inside the `srt-anom` environment without requiring manual activation.

The pipeline entry point is `src/srtad/main.py`. Pipeline behaviour (paths, hyperparameters, first-stage filter, thresholds) is controlled by `config/default.yaml`.

---

## Generated outputs

All artifacts are written under the runtime `results/` directory:

- `passed_density.csv` — candidates promoted by the Density Filter.
- `passed_autoencoder.csv` — candidates promoted by the Autoencoder Filter.
- `top_frequency_percentile.csv` — top frequency-score percentile per star–band combination.
- `top_similarity_percentile.csv` — top similarity-score percentile per star–band combination.
- `top_k_combined.csv` — top-K candidates by combined frequency and similarity scores.
- `random_control.csv` — random control sample for statistical comparison.
- `REPORT_CATEGORIES.pdf` — diagnostic category report for manual inspection.

---

## Reference

S. Pardo, D. Poznanski, S. Croft, A. P. Siemion, and M. Lebofsky,
*Using anomaly detection to search for technosignatures in Breakthrough Listen observations*,
The Astronomical Journal, vol. 170, no. 1, p. 12, 2025.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
