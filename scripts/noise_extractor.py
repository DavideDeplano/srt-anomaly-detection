"""
Extract a random background noise patch from a real .h5 waterfall file.

This script:
- Loads a real Breakthrough Listen .h5 file
- Extracts a 2D time-frequency slice
- Randomly crops a (tchans x fchans) region
- Applies the same preprocessing used in the main pipeline
- Saves the resulting noise patch to disk for reuse
"""

from pathlib import Path
import sys
import numpy as np
from blimpy import Waterfall
from srtad.config import paths, simulation, random_seed
from srtad.core.dataset import Dataset

# Input .h5 file
H5_PATH = Path("content/hdd_esterno/TIC_H5/guppi_60703_21005_031211_TIC168699373_OFF_0001.0000.h5")

# Parameters from configuration
# Fixed random seed to ensure reproducible noise extraction
seed = int(random_seed)

# Target patch size (must match pipeline expectations)
tchans = int(simulation["tchans"])   # 16
fchans = int(simulation["fchans"])   # 80

# Output path for the extracted background noise patch
data_root = Path(paths["data"])
out_path = data_root / "noise" / "background_noise.npy"
out_path.parent.mkdir(parents=True, exist_ok=True)

# Load .h5 file and reduce to a 2D (time, frequency) array
wf = Waterfall(str(H5_PATH), load_data=True)

# Raw data may have shape (time, freq), (time, pol, freq), or (pol, time, freq)
arr = np.asarray(wf.data)
arr = np.squeeze(arr)

if arr.ndim == 2:
    # Already in (time, frequency) form
    data2d = arr
elif arr.ndim == 3:
    # Handle common polarization layouts
    if arr.shape[1] in (1, 2, 4):
        # Shape: (time, pol, freq) → take first polarization
        data2d = arr[:, 0, :]
    elif arr.shape[0] in (1, 2, 4):
        # Shape: (pol, time, freq) → take first polarization
        data2d = arr[0, :, :]
    else:
        raise ValueError(f"Unsupported .h5 shape after squeeze: {arr.shape}")
else:
    raise ValueError(f"Unsupported .h5 shape after squeeze: {arr.shape}")

# Ensure numerical stability and consistent dtype
data2d = np.asarray(data2d, dtype=np.float64)

# Validate minimum size for random cropping
T, F = data2d.shape
if T < tchans or F < fchans:
    raise ValueError(f"Input too small: {data2d.shape}, need at least ({tchans},{fchans}).")

# Random crop extraction (background noise patch)
rng = np.random.default_rng(seed)

# Reproducible random generator
t0 = int(rng.integers(0, T - tchans + 1))
f0 = int(rng.integers(0, F - fchans + 1))

# Apply internal spectrogram preprocessing 
noise = data2d[t0:t0 + tchans, f0:f0 + fchans]

# Preprocessing (same pipeline used for real/simulated data)
ds = Dataset()

# Apply internal spectrogram preprocessing
noise = ds._preprocess_spectrogram(noise)

# Replace NaN / Inf values to ensure safe downstream usage
noise = np.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)

# Persist extracted background noise
np.save(out_path, noise)
