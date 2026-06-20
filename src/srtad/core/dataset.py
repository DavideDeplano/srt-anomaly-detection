"""
Dataset management class.

Handles:
- loading and packaging of synthetic cadence data;
- loading, cropping, and packaging of real PNG waterfall plots
  into Candidate objects for downstream analysis.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from skimage.transform import resize
from joblib import Parallel, delayed
import re
import logging
import numpy as np
import csv

from src.srtad.core.candidate import Candidate

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# ── module-level regex (compiled once, reused by worker processes) ────────────

_RX_CANDIDATE = re.compile(
    r"^candidate_\d+_(?P<freq>[\d.]+)MHz\.png$",
    re.IGNORECASE,
)
_RX_GENERIC = re.compile(
    r"(?:_dr_(?P<dr>[-+0-9.eE]+))?_freq_(?P<freq1>[-+0-9.eE]+)|_(?P<freq2>[\d.]+)MHz",
    re.IGNORECASE,
)
_RX_TIC = re.compile(r"(?:^|[_\-])(?P<tic>TIC\d+)(?:[_\-]|$)", re.IGNORECASE)

TARGET_H = 16
TARGET_W = 80


def _find_tic_in_parents(path: Path) -> Optional[str]:
    """Walk up the directory hierarchy looking for a folder named TIC<digits>."""
    for parent in path.parents:
        if re.match(r"^TIC\d+$", parent.name, re.IGNORECASE):
            return parent.name
    return None


def _find_separator_rows(arr: np.ndarray, white_threshold: float = 250) -> list:
    """Find horizontal white separator rows."""
    row_means = arr.mean(axis=1)
    white_rows = np.where(row_means > white_threshold)[0]
    
    if len(white_rows) == 0:
        return []
    
    separators = []
    current_group = [white_rows[0]]
    
    for i in range(1, len(white_rows)):
        if white_rows[i] - white_rows[i-1] <= 3:
            current_group.append(white_rows[i])
        else:
            if len(current_group) >= 3:
                separators.append(int(np.median(current_group)))
            current_group = [white_rows[i]]
    
    if current_group and len(current_group) >= 3:
        separators.append(int(np.median(current_group)))
    
    return separators


def _crop_box(w: int, h: int, is_candidate_format: bool) -> Tuple[int, int, int, int]:
    """Compute crop rectangle as fixed fractions of image width/height."""
    if is_candidate_format:
        return (
            int(w * 0.0461),
            int(h * 0.1163),
            int(w * (1 - 0.086)),
            int(h * (1 - 0.04)),
        )
    else:
        return (
            int(w * 0.0894),
            int(h * 0.12),
            int(w * (1 - 0.149)),
            int(h * (1 - 0.03)),
        )


def _preprocess_spectrogram(data: np.ndarray) -> np.ndarray:
    """
    Apply normalization/cleaning steps to a 2D spectrogram array.

    Steps:
    1) Time normalization
    2) DC spike removal
    3) Bandpass correction via spline + ridge regression
    """
    data = np.asarray(data, dtype=np.float64)
    data = np.maximum(data, 1e-9)

    time_means = np.mean(data, axis=1, keepdims=True)
    data = data / np.maximum(time_means, 1e-9)

    H, W = data.shape
    dc_index = W // 2
    if 0 < dc_index < W - 1:
        data[:, dc_index] = (data[:, dc_index - 1] + data[:, dc_index + 1]) / 2.0

    bandpass = np.mean(data, axis=0)
    X = np.arange(W, dtype=np.float64).reshape(-1, 1)
    y = bandpass.astype(np.float64)

    try:
        model = make_pipeline(
            SplineTransformer(n_knots=20, degree=3, include_bias=False),
            Ridge(alpha=1.0),
        )
        model.fit(X, y)
        smooth_bandpass = model.predict(X).astype(np.float64)
        smooth_bandpass = np.nan_to_num(smooth_bandpass, nan=1.0, posinf=1.0, neginf=1.0)
        smooth_bandpass = np.maximum(smooth_bandpass, 1e-9)
        data = data / smooth_bandpass.reshape(1, -1)
    except Exception:
        pass

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def _minmax_panel(data: np.ndarray) -> np.ndarray:
    """
    Per-panel min-max normalization to [0, 1], WITHOUT bandpass correction.

    Used to build the similarity-filter representation. The bandpass step in
    `_preprocess_spectrogram` divides out persistent narrowband lines (which is
    exactly the morphology of a non-drifting candidate); this normalization
    preserves them.
    """
    q = np.asarray(data, dtype=np.float64)
    q = q - q.min()
    m = q.max()
    if m > 0:
        q = q / m
    return q


def _load_single_png(png: Path):
    """
    Load, crop, preprocess and resize a single PNG into a Candidate.

    Returns a Candidate on success, or None on failure/skip.
    This function is module-level so joblib can pickle it for multiprocessing.
    """
    is_candidate_format = bool(_RX_CANDIDATE.match(png.name))

    if is_candidate_format:
        m = _RX_CANDIDATE.match(png.name)
        freq_hz = float(m.group("freq")) * 1e6
        drift_hz_s = 0.0
    else:
        m = _RX_GENERIC.search(png.name)
        if not m:
            return None, str(png)  # skipped
        if m.group("freq1") is not None:
            freq_hz = float(m.group("freq1").strip(".")) * 1e6
            drift_hz_s = float(m.group("dr")) if m.group("dr") else 0.0
        else:
            freq_hz = float(m.group("freq2").strip(".")) * 1e6
            drift_hz_s = 0.0

    # Target extraction
    tic_match = _RX_TIC.search(png.name)
    if tic_match:
        target = tic_match.group("tic")
    else:
        target = _find_tic_in_parents(png) or png.parent.name

    # Notch filter
    if (1.2e9 <= freq_hz <= 1.33e9) or (2.3e9 <= freq_hz <= 2.36e9):
        return None, None  # silently skipped

    # Load image
    try:
        with Image.open(png) as im:
            im = im.convert("L")
            w, h = im.size
            arr = np.asarray(im, dtype=np.float64)
    except Exception:
        return None, None

    panels = []

    # ========== CANDIDATE FORMAT (h5_to_candidates.py: NO white separators) ==========
    if is_candidate_format:
        H, W = arr.shape
        
        # Global crop
        left   = int(W * 0.0894)
        right  = int(W * (1 - 0.149))
        top    = int(H * 0.12)
        bottom = int(H * (1 - 0.03))
        
        cropped = arr[top:bottom, left:right]
        CH, _ = cropped.shape
        
        # Uniform split (no white separators)
        step = CH // 6
        if step == 0:
            return None, None
        
        for i in range(6):
            panel = cropped[i * step:(i + 1) * step, :]
            panels.append(panel)

    # ========== TIC FORMAT: separator detection with uniform-split fallback ==========
    else:
        H, W = arr.shape
        
        # Global crop
        left   = int(W * 0.12)
        right  = int(W * 0.88)
        top    = int(H * 0.04)
        bottom = int(H * 0.95)
        
        cropped = arr[top:bottom, left:right]
        CH, _ = cropped.shape
        
        # Find separators
        separators = _find_separator_rows(cropped, white_threshold=250)
        
        if len(separators) >= 5:
            # ---- Filippo's plots: white separator bands between panels ----
            separators = separators[:5]
            boundaries = [0] + separators + [CH]
            
            # Extract panels, skipping white bands
            for i in range(6):
                start = boundaries[i]
                end = boundaries[i+1]
                
                if i > 0:
                    start += 25
                if i < 5:
                    end -= 25
                
                if end <= start:
                    return None, None
                
                panel = cropped[start:end, :]
                panels.append(panel)
        else:
            # ---- h5_to_candidates.py plots: adjacent panels, no separators ----
            # Re-crop with margins measured for this layout (excludes title,
            # axis labels, and colorbar)
            left   = int(W * 0.0517)
            right  = int(W * (1 - 0.0827))
            top    = int(H * 0.1179)
            bottom = int(H * (1 - 0.0296))
            
            cropped = arr[top:bottom, left:right]
            CH, _ = cropped.shape
            
            step = CH // 6
            if step == 0:
                return None, None
            
            for i in range(6):
                panel = cropped[i * step:(i + 1) * step, :]
                panels.append(panel)

    # ========== COMMON: build TWO representations ==========
    # Keep the raw cropped panels for the min-max (similarity) representation.
    panels_crop = [np.asarray(p, dtype=np.float64) for p in panels]

    def _resize_panel(p: np.ndarray) -> np.ndarray:
        return resize(
            p,
            (TARGET_H, TARGET_W),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )

    # (A) bandpassed (Pardo-style) -> used by density / frequency filters.
    #     Behavior unchanged with respect to the previous pipeline.
    try:
        panels_bp = [_preprocess_spectrogram(p) for p in panels_crop]
    except Exception:
        return None, None

    min_h = min(p.shape[0] for p in panels_bp)
    panels_bp = [p[:min_h, :] for p in panels_bp]

    try:
        cadence = np.stack([_resize_panel(p) for p in panels_bp], axis=0)
    except ValueError:
        return None, None

    # (B) per-panel min-max, NO bandpass -> used by the similarity filter.
    min_h2 = min(p.shape[0] for p in panels_crop)
    panels_mm = [_minmax_panel(p[:min_h2, :]) for p in panels_crop]

    try:
        cadence_raw = np.stack([_resize_panel(p) for p in panels_mm], axis=0)
    except ValueError:
        return None, None

    candidate = Candidate(
        id=png.stem,
        frequency_hz=freq_hz,
        drift_hz_s=drift_hz_s,
        cadence=cadence,
        source_path=png,
    )
    candidate.set_cadence_raw(cadence_raw)
    candidate.set_target(target if target else "UNKNOWN")
    return candidate, None


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

        self._rx_candidate = _RX_CANDIDATE
        self._rx_generic = _RX_GENERIC
        self._rx_tic = _RX_TIC

    @staticmethod
    def _find_tic_in_parents(path: Path) -> Optional[str]:
        """Walk up the directory hierarchy looking for a folder named TIC<digits>."""
        return _find_tic_in_parents(path)

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

        for cid in metadata_index:
            metadata_index[cid]["panels"].sort(key=lambda d: d["slot"])

        cadences: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []

        cid_iter = metadata_index.items()
        if self._use_tqdm and tqdm is not None:
            cid_iter = tqdm(cid_iter, desc="Loading synthetic cadence tensors")

        for cadence_id, meta in cid_iter:
            tensor_path = cadences_dir / f"{cadence_id}.npy"
            if not tensor_path.exists():
                raise FileNotFoundError(f"Missing cadence tensor: {tensor_path}")

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

    def _crop_box(self, w: int, h: int, is_candidate_format: bool) -> Tuple[int, int, int, int]:
        """Compute the crop rectangle as fixed fractions of image width/height."""
        return _crop_box(w, h, is_candidate_format)

    def _preprocess_spectrogram(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization/cleaning steps to a 2D spectrogram array."""
        return _preprocess_spectrogram(data)

    def load(self, png_dir: str | Path | None = None) -> List[Candidate]:
        """
        Load real PNG candidates from disk and convert them into Candidate objects.

        Uses joblib.Parallel for parallel loading across all available CPU cores.
        """
        search_dir = Path(png_dir) if png_dir is not None else self._png_dir

        if not search_dir.exists():
            self._logger.warning("Data path not found: %s", search_dir)
            return []

        png_files = sorted(search_dir.rglob("*.png"))
        self._logger.info("Found %d PNG files in %s", len(png_files), search_dir)

        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_load_single_png)(png)
            for png in tqdm(png_files, desc="Loading real PNG candidates")
        )

        candidates: List[Candidate] = []
        skipped_files: List[str] = []

        for candidate, skipped in results:
            if candidate is not None:
                candidates.append(candidate)
            elif skipped is not None:
                skipped_files.append(skipped)

        if skipped_files:
            skipped_path = Path("results") / "skipped_files.txt"
            skipped_path.parent.mkdir(parents=True, exist_ok=True)
            with open(skipped_path, "w") as f:
                f.write(f"Skipped {len(skipped_files)} files with unexpected filename format:\n\n")
                for p in skipped_files:
                    f.write(f"{p}\n")
            self._logger.warning(
                "%d files skipped due to unexpected filename format. See: %s",
                len(skipped_files),
                skipped_path,
            )

        self._logger.info("Loaded %d real PNG candidates from %s", len(candidates), search_dir)
        return candidates