#!/usr/bin/env bash
# Exit immediately on error (-e), treat unset variables as errors (-u),
# and make pipelines fail if any command fails (-o pipefail)
set -euo pipefail

# Execute the main module inside the 'srt-anom' environment
# "$@" forwards any command-line arguments to the Python program
conda run -n srt-anom python -m src.srtad.main "$@"
