#!/usr/bin/env bash
# Exit immediately on error (-e), treat unset variables as errors (-u),
# and make pipelines fail if any command fails (-o pipefail)
set -euo pipefail

# Create or update the Conda environment from env.yml
conda env create -f env.yml || conda env update -f env.yml

# Install the local project in editable mode inside the environment
conda run -n srt-anom python -m pip install -e .

echo "[OK] Environment ready: srt-anom"
