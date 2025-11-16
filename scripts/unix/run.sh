#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment (interactive safe)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate srt-anom

# Run main module normally (interactive OK)
python -m src.srtad.main "$@"
