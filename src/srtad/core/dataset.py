"""
Dataset management class.

Handles:
- loading and packaging of synthetic cadence data;
- loading, cropping, and packaging of real PNG waterfall plots
  into Candidate objects for downstream analysis.
"""

from pathlib import Path
import re
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import csv
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from .candidate import Candidate
from skimage.transform import resize

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

class Dataset:
    """
    Dataset wrapper.

    Responsibilities:
    - Load real PNG files, preprocess them, split them into 6 panels, and
      package them into Candidate objects.
    - Load synthetic cadences from .npy files and attach metadata derived
      from the associated cadences_log.csv file.
    """

    def __init__(self, png_dir: str | Path | None = None, use_tqdm: bool = False) -> None:
        """
        Parameters
        ----------
        png_dir:
            Root directory containing PNG waterfall plots.
            If None, defaults to "data/SRT_dataset".
        use_tqdm:
            Enable tqdm progress bars when iterating files.
        """
        self._png_dir = Path(png_dir) if png_dir is not None else (Path("data") / "SRT_dataset")
        self._logger = logging.getLogger("srtad.dataset")
        self._use_tqdm = bool(use_tqdm)

        # Regex used to parse drift rate and frequency from the PNG filename
        # Expected pattern: "..._dr_<float>_freq_<float>.png"
        self._rx = re.compile(
            r'^.*_dr_(?P<dr>[-+0-9.eE]+)_freq_(?P<freq>[-+0-9.eE]+)\.png$',
            re.IGNORECASE
        )


    def load_simulated_cadences(
        self,
        cadences_dir: Path | str,
    ) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        """
        Load synthetic cadence tensors and their metadata.

        Inputs
        ------
        - Tensor files: "<cadence_id>.npy" where cadence_id matches the keys
          built from the CSV log.
        - Metadata log: "cadences_log.csv" located inside cadences_dir.

        Expected directory structure
        ----------------------------
            cadences_dir/
                cadences_log.csv
                cadence_00000_pattern00.npy
                cadence_00000_pattern01.npy
                ...

        Returns
        -------
        List of (cadence_id, tensor, metadata_dict), where:
        - cadence_id: e.g. "cadence_00042_pattern03"
        - tensor: np.ndarray with shape (6, H, W)
        - metadata_dict:
            - "pattern_id": integer pattern identifier
            - "panels": list of per-slot dictionaries (sorted by slot index)
        """
        cadences_dir = Path(cadences_dir)
        log_path = cadences_dir / "cadences_log.csv"

        if not log_path.exists():
            raise FileNotFoundError(f"cadences_log.csv not found in {cadences_dir}")

        self._logger.info("Loading synthetic cadences from %s", cadences_dir)

        # Build an index: cadence_id -> metadata dictionary
        metadata_index: Dict[str, Dict[str, Any]] = {}

        with open(log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cadence_base = row["cadence_id"]              
                pattern_id = int(row["pattern_id"])           
                cid = f"{cadence_base}_pattern{pattern_id:02d}"  

                if cid not in metadata_index:
                    metadata_index[cid] = {
                        "pattern_id": pattern_id,
                        "panels": [],
                    }

                # Store per-slot metadata from the CSV row
                metadata_index[cid]["panels"].append(
                    {
                        "slot": int(row["slot"]),
                        "on": bool(int(row["on"])),
                        "amplitude_factor": float(row["amplitude_factor"]),
                        "drift_rate_hz_s": float(row["drift_rate_hz_s"]),
                        "width_hz": float(row["width_hz"]),
                        "f_start_mhz": float(row["f_start_mhz"]),
                        "f_start_idx": int(row["f_start_idx"]),
                        "tchans": int(row["tchans"]),
                        "fchans": int(row["fchans"]),
                        "df_hz": float(row["df_hz"]),
                        "dt_s": float(row["dt_s"]),
                        "fch1_mhz": float(row["fch1_mhz"]),
                        "random_seed": int(row["random_seed"]),
                    }
                )

        # Ensure panel metadata is ordered by slot index
        for cid in metadata_index:
            metadata_index[cid]["panels"].sort(key=lambda d: d["slot"])

        cadences: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []

        # Iterate over all cadence IDs from the metadata index and load the .npy tensors
        cid_iter = metadata_index.items()
        if self._use_tqdm and tqdm is not None:
            cid_iter = tqdm(cid_iter, desc="Loading synthetic cadence tensors")

        for cadence_id, meta in cid_iter:
            tensor_path = cadences_dir / f"{cadence_id}.npy"
            if not tensor_path.exists():
                raise FileNotFoundError(f"Missing cadence tensor: {tensor_path}")

            # Load tensor and apply preprocessing panel-by-panel (6 panels expected)
            raw_tensor = np.load(tensor_path)
            preprocessed_panels = []
            for i in range(6):
                panel = raw_tensor[i, :, :]
                clean_panel = self._preprocess_spectrogram(panel)
                preprocessed_panels.append(clean_panel)
            
            clean_tensor = np.stack(preprocessed_panels, axis=0)
            cadences.append((cadence_id, clean_tensor, meta))

        self._logger.info("Loaded %d synthetic cadences from %s", len(cadences), cadences_dir)
        return cadences

    def _crop_box(self, w: int, h: int) -> tuple[int, int, int, int]:
        """
        Compute the crop rectangle as fixed fractions of image width/height.

        This removes fixed margins from the PNG image before further processing.
        """
        return (
            int(w * 0.0894),        # left margin
            int(h * 0.044),         # top margin
            int(w * (1 - 0.149)),   # right margin
            int(h * (1 - 0.067)),   # bottom margin
        )
    
    def _preprocess_spectrogram(self, data: np.ndarray) -> np.ndarray:
        """
        Apply normalization/cleaning steps to a 2D spectrogram array.

        Steps performed:
        1) Time normalization: each time row is divided by its mean over frequency.
        2) DC spike removal: the central frequency channel is replaced with the
           average of its immediate neighbors.
        3) Bandpass correction: divide by a smoothed bandpass estimate obtained
           via a spline transformer + ridge regression pipeline.

        Notes:
        - The bandpass smoothing is best-effort: on failure, the input data is
          left as-is after steps (1) and (2), and a warning is logged.
        - Final outputs are forced to finite values via np.nan_to_num.
        """
        # Convert to float and avoid zeros
        data = np.asarray(data, dtype=np.float64)
        data = np.maximum(data, 1e-9)

        # 1) Time normalization: normalize each row by its frequency-mean
        time_means = np.mean(data, axis=1, keepdims=True)          
        data = data / np.maximum(time_means, 1e-9)

        # 2) DC spike removal: replace center channel with neighbor average
        H, W = data.shape
        dc_index = W // 2
        if 0 < dc_index < W - 1:
            data[:, dc_index] = (data[:, dc_index - 1] + data[:, dc_index + 1]) / 2.0

        # 3) Bandpass correction using a spline + ridge regression model
        bandpass = np.mean(data, axis=0)                           

        X = np.arange(W, dtype=np.float64).reshape(-1, 1)
        y = bandpass.astype(np.float64)

        try:
            model = make_pipeline(
                SplineTransformer(n_knots=20, degree=3, include_bias=False),
                Ridge(alpha=1.0),
            )
            model.fit(X, y)
            smooth_bandpass = model.predict(X).astype(np.float64)  # shape (W,)

            # Safety: prevent division by tiny/invalid values
            smooth_bandpass = np.nan_to_num(smooth_bandpass, nan=1.0, posinf=1.0, neginf=1.0)
            smooth_bandpass = np.maximum(smooth_bandpass, 1e-9)

            data = data / smooth_bandpass.reshape(1, -1)

        except Exception as e:
            self._logger.warning(f"Scikit-learn B-spline fitting failed: {e}")

        # Final safety: enforce finite output values.
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data

    def load(self, png_dir: str | Path | None = None) -> List[Candidate]:
        """
        Load real PNG candidates from disk and convert them into Candidate objects.

        For each PNG file:
        - Parse drift and frequency from filename via regex.
        - Skip files in the hard-coded notch ranges.
        - Open the image, convert to grayscale, crop margins, and preprocess.
        - Split vertically into 6 panels using integer division of the height.
        - Resize each panel to a fixed shape (TARGET_H, TARGET_W).
        - Stack panels into a cadence tensor of shape (6, TARGET_H, TARGET_W).
        """
        search_dir = Path(png_dir) if png_dir is not None else self._png_dir

        if not search_dir.exists():
            self._logger.warning("Data path not found: %s", search_dir)
            return []

        candidates: List[Candidate] = []

        png_iter = sorted(search_dir.rglob("*.png"))
        if self._use_tqdm and tqdm is not None:
            png_iter = tqdm(png_iter, desc="Loading real PNG candidates")

        for png in png_iter:
            match = self._rx.match(png.name)
            if not match:
                self._logger.debug("Skipping file with unexpected name: %s", png.name)
                continue

            # Parse drift and frequency values from the filename
            drift_hz_s = float(match.group("dr"))
            freq_hz = float(match.group("freq")) * 1e6  # MHz -> Hz

            # Skip candidates falling into hard-coded notch frequency ranges
            if (1.2e9 <= freq_hz <= 1.33e9) or (2.3e9 <= freq_hz <= 2.36e9):
                self._logger.info(
                    "Skipping candidate in notch filter range: %.6f MHz", freq_hz / 1e6
                )
                continue

            # Load image, convert to grayscale, crop fixed margins, and preprocess
            try:
                with Image.open(png) as im:
                    im = im.convert("L")  # grayscale: (H, W)
                    w, h = im.size
                    cropped = im.crop(self._crop_box(w, h))
                    arr = np.asarray(cropped, dtype=np.float64)
                    arr = self._preprocess_spectrogram(arr)
            except Exception as exc:
                self._logger.error("Failed to load or process image %s: %s", png, exc)
                continue

            # Split the processed image into 6 vertical panels
            # Note: step is computed via integer division; if H is not divisible by 6,
            # the remainder rows are not included in the panels
            H = arr.shape[0]
            step = H // 6
            if step == 0:
                self._logger.warning(
                    "Image too small to split into 6 panels: %s (H=%d)", png, H
                )
                continue

            panels = [arr[i * step:(i + 1) * step, :] for i in range(6)]

            # Fixed output size for each panel
            TARGET_H = 16
            TARGET_W = 80

            # Align panel heights to a common value before resizing
            min_h = min(p.shape[0] for p in panels)
            panels = [p[:min_h, :] for p in panels]

            # Resize each panel to (TARGET_H, TARGET_W)
            panels_resized = [
                resize(
                    p,
                    (TARGET_H, TARGET_W),
                    order=1,
                    mode="reflect",
                    anti_aliasing=True,
                    preserve_range=True
                )
                for p in panels
            ]

            # Stack resized panels into a cadence tensor: (6, TARGET_H, TARGET_W)
            try:
                cadence = np.stack(panels_resized, axis=0)  # shape: (6, H, W)
            except ValueError as exc:
                self._logger.error(
                    "Failed to stack panels into cadence for %s: %s", png, exc
                )
                continue

            # Create a Candidate using the filename stem as identifier
            candidate = Candidate(
                id=png.stem,
                frequency_hz=freq_hz,
                drift_hz_s=drift_hz_s,
                cadence=cadence,
                source_path=png,
            )
            candidates.append(candidate)

        self._logger.info("Loaded %d real PNG candidates from %s", len(candidates), self._png_dir)
        return candidates

   