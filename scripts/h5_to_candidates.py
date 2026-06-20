"""
h5_to_candidates.py

Scan a directory of SRT .h5 files, group them into cadences
(6 ON/OFF observations per cadence), slide a window of VIZ_CHANS channels
along the frequency axis, and save each window as a PNG in real_png_dir.

Cadences are interleaved by frequency band (C/K) for balanced output.

Usage
-----
    python scripts/h5_to_candidates.py --scan /path/to/h5_dir --dry-run
    python scripts/h5_to_candidates.py --scan /path/to/h5_dir
    python scripts/h5_to_candidates.py --scan /path/to/h5_dir --skip-cadences 107
    python scripts/h5_to_candidates.py --scan /path/to/h5_dir --output /path/to/dir --step 4096
"""

import argparse
import re
import warnings
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.srtad.config import paths, simulation as sim_cfg

try:
    from blimpy import Waterfall
except ImportError:
    raise ImportError("blimpy is required: pip install blimpy")

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# Parameters
N_PANELS       = int(sim_cfg["panels_per_cadence"])  # 6
VIZ_CHANS      = 4096                                # channels per PNG window
DOWNSAMPLE     = 8                                    # frequency downscale factor
FINAL_FREQ_BINS = VIZ_CHANS // DOWNSAMPLE             # 512

ON_OFF_PATTERN = ["ON", "OFF", "ON", "OFF", "ON", "OFF"]

BAND_RANGES = {
    "C": (4200,  7700),
    "K": (18000, 26500),
}


# ── band detection ────────────────────────────────────────────────────────────

def _detect_band(file_path: Path) -> str:
    """Detect frequency band from .h5 header without loading data."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf = Waterfall(str(file_path), load_data=False)
        fch1 = float(wf.header.get("fch1", 0.0))
        for band, (fmin, fmax) in BAND_RANGES.items():
            if fmin <= fch1 <= fmax:
                return band
        return "unknown"
    except Exception:
        return "unknown"


# ── preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_cadence(cadence: np.ndarray) -> np.ndarray:
    """
    Per-observation log min-max normalization [0, 1] at full resolution.

    Each observation panel is log-transformed and normalized independently
    on its own min/max. The frequency axis is NOT downscaled.

    Input:  (6, time, freq)  e.g. (6, 16, 4096)
    Output: (6, time, freq)  normalized [0, 1]
    """
    data = cadence.astype(np.float64)

    # Log-normalize PER OBSERVATION at full channel resolution.
    # No frequency downscale: averaging channels destroys the fine-scale
    # structure shared between panels (persistent narrowband lines, bandpass
    # ripple) that the cross-correlation features rely on.
    out = np.zeros_like(data)
    for i in range(data.shape[0]):
        obs = np.log(np.abs(data[i]) + 1e-10)
        obs = obs - obs.min()
        max_val = obs.max()
        if max_val > 0:
            obs = obs / max_val
        out[i] = obs

    return out


# ── file parsing and grouping ─────────────────────────────────────────────────

def _extract_timestamp(filepath: Path) -> int:
    """Extract timestamp from filename for cadence ordering."""
    match = re.search(r"guppi_(\d+)_(\d+)_", filepath.name)
    if match:
        return int(match.group(1)) * 1_000_000 + int(match.group(2))
    return 0


def _cadence_tag(filepath: Path) -> str:
    """
    Unique tag for a cadence, derived from the timestamp of its first
    observation file. Used in PNG filenames to disambiguate multiple
    cadences of the same target.
    """
    match = re.search(r"guppi_(\d+)_(\d+)_", filepath.name)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return "00000_00000"


def _parse_filename(filepath: Path) -> Optional[Dict]:
    """Parse .h5 filename to extract target, observation type, and metadata."""
    name = filepath.stem
    tic_match = re.search(r"(TIC\d+)_(ON|OFF)", name)
    if not tic_match:
        return None
    time_match = re.search(r"_(\d{5})_(\d+)_", name)
    mjd = time_match.group(1) if time_match else "unknown"
    parent = filepath.parent.name
    date_match = re.search(r"(\d{8})", parent)
    date = date_match.group(1) if date_match else mjd
    return {
        "target":    tic_match.group(1),
        "obs_type":  tic_match.group(2),
        "date":      date,
        "timestamp": _extract_timestamp(filepath),
        "filepath":  filepath,
    }


def group_into_cadences(files: List[Path]) -> List[Tuple[str, List[Path], str]]:
    """
    Group .h5 files into valid cadences (ON OFF ON OFF ON OFF).

    Returns
    -------
    list of (target_name, [6 file paths], band)
    """
    groups = defaultdict(list)
    for f in files:
        info = _parse_filename(f)
        if info:
            key = f"{info['target']}_{info['date']}_{f.parent}"
            groups[key].append(info)

    cadences = []
    skipped = 0
    for key, infos in groups.items():
        infos.sort(key=lambda x: x["timestamp"])
        if len(infos) != N_PANELS:
            skipped += 1
            continue
        if [i["obs_type"] for i in infos] != ON_OFF_PATTERN:
            skipped += 1
            continue
        band = _detect_band(infos[0]["filepath"])
        cadences.append((infos[0]["target"], [i["filepath"] for i in infos], band))

    if skipped:
        print(f"  [WARN] Skipped {skipped} incomplete/invalid cadences")
    return cadences


def interleave_by_band(cadences: List[Tuple]) -> List[Tuple]:
    """Interleave cadences by band so output is balanced across C and K."""
    by_band = defaultdict(list)
    for item in cadences:
        by_band[item[2]].append(item)

    interleaved = []
    iters = {b: iter(v) for b, v in by_band.items()}
    while iters:
        exhausted = []
        for b, it in iters.items():
            try:
                interleaved.append(next(it))
            except StopIteration:
                exhausted.append(b)
        for b in exhausted:
            del iters[b]
    return interleaved


# ── PNG saving ────────────────────────────────────────────────────────────────

def _save_png(cadence_tensor: np.ndarray,
              target: str,
              freq_mhz: float,
              output_dir: Path,
              candidate_idx: int = None,
              filename: str = None,
              cadence_tag: str = None) -> Path:
    """
    Save cadence as PNG.

    Parameters
    ----------
    cadence_tensor : np.ndarray
        Preprocessed cadence data (6, time, freq_bins) — normalized [0, 1].
        Frequency axis must be increasing left to right (flip applied upstream
        when foff < 0).
    target : str
        Target name
    freq_mhz : float
        Center frequency in MHz
    output_dir : Path
        Output directory
    candidate_idx : int, optional
        Candidate index for filename (candidate_N_FREQ.png)
    filename : str, optional
        Custom filename (overrides default naming)
    cadence_tag : str, optional
        Unique cadence tag included in the default filename to disambiguate
        multiple cadences of the same target
    """
    from src.srtad.config import simulation as _sim
    dt_s  = float(_sim["dt_s"])
    df_hz = float(_sim["df_hz"])

    # Frequency axis at raw channel resolution
    channel_width_hz = df_hz
    half_bins = cadence_tensor.shape[2] // 2
    freq_axis_hz = np.arange(-half_bins, half_bins) * channel_width_hz

    # Time axis
    time_axis = np.arange(cadence_tensor.shape[1]) * dt_s

    fig, axes = plt.subplots(N_PANELS, 1, figsize=(14, 14), sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle(f"{target} @ {freq_mhz:.6f} MHz", fontsize=14, fontweight="bold")

    imgs = []
    for i, ax in enumerate(axes):
        im = ax.imshow(
            cadence_tensor[i],
            aspect="auto",
            origin="upper",
            cmap="viridis",
            interpolation="bilinear",
            extent=[freq_axis_hz[0], freq_axis_hz[-1], time_axis[-1], time_axis[0]],
            vmin=0, vmax=1,
        )
        imgs.append(im)
        ax.set_ylabel("Time [s]", fontsize=8)
        if i < N_PANELS - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(f"Relative Frequency [Hz] from {freq_mhz:.6f} MHz", fontsize=9)

    # Limit the displayed span to +/-2861 Hz (2048 raw channels), matching
    # the field of view measured on the Zuddas reverse-search PNGs.
    # Clamped to the actual window span for narrower windows.
    half_span = min(2861.0, float(abs(freq_axis_hz[0])))
    for ax in axes:
        ax.set_xlim(-half_span, half_span)

    cbar = fig.colorbar(imgs[0], ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("Power", fontsize=10)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.patch.set_facecolor("white")

    if filename:
        fname = filename
    elif cadence_tag:
        fname = f"{target}_{cadence_tag}_{freq_mhz:.6f}MHz.png"
    else:
        fname = f"{target}_{freq_mhz:.6f}MHz.png"

    out_path = output_dir / fname
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ── cadence processing ────────────────────────────────────────────────────────

def process_cadence(target: str,
                    file_paths: List[Path],
                    output_dir: Path,
                    start_idx: int,
                    step: int,
                    max_windows: Optional[int] = None) -> int:
    """
    Process a single cadence: load observations, slide window, save PNGs.

    Returns
    -------
    Number of PNGs saved for this cadence.
    """
    obs_data   = []
    freq_axis  = None
    foff       = 0.0

    for fp in file_paths:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf = Waterfall(str(fp))
        data = wf.data.squeeze()
        obs_data.append(data.astype(np.float32))

        if freq_axis is None:
            header    = wf.header
            fch1      = header.get("fch1", 0.0)
            foff      = float(header.get("foff", 0.0))
            nchans    = data.shape[1]
            freq_axis = fch1 + foff * np.arange(nchans)

    cadence_array = np.stack(obs_data, axis=0)
    n_freq        = cadence_array.shape[2]

    # Unique tag for this cadence (timestamp of its first observation):
    # disambiguates filenames when the same target has multiple cadences.
    tag = _cadence_tag(file_paths[0])

    n_saved = 0
    idx     = start_idx

    for start in range(0, n_freq - VIZ_CHANS + 1, step):
        if max_windows is not None and n_saved >= max_windows:
            break

        end      = start + VIZ_CHANS
        # Frequency of the window centre channel (matches Zuddas pipeline)
        center_chan = start + VIZ_CHANS // 2
        freq_mhz = float(freq_axis[center_chan])

        viz_window = cadence_array[:, :, start:end]

        # If foff < 0, channel index runs from high to low frequency:
        # flip so that the x axis increases with frequency
        if foff < 0:
            viz_window = viz_window[:, :, ::-1]

        processed = _preprocess_cadence(viz_window)

        _save_png(processed, target, freq_mhz, output_dir,
                  candidate_idx=idx, cadence_tag=tag)
        idx     += 1
        n_saved += 1

    return n_saved


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(scan_dirs: List[str]) -> None:
    """Print statistics about what would be generated without creating files."""
    all_files: List[Path] = []
    for d in scan_dirs:
        all_files.extend(Path(d).rglob("*.h5"))

    if not all_files:
        print("[ERROR] No .h5 files found.")
        return

    first = all_files[0]
    print(f"\nDry run on: {first.name}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf = Waterfall(str(first), load_data=False)

    header   = wf.header
    n_freq   = header.get("nchans", 0)
    fch1     = header.get("fch1", 0.0)
    foff     = header.get("foff", 0.0)
    freq_end = fch1 + foff * n_freq

    print(f"  Channels  : {n_freq:,}")
    print(f"  Freq range: {min(fch1, freq_end):.2f} – {max(fch1, freq_end):.2f} MHz")
    print(f"  VIZ_CHANS : {VIZ_CHANS}")

    print("\n  Grouping cadences (this may take a moment — reads headers)...")
    cadences = group_into_cadences(all_files)
    print(f"  Cadences  : {len(cadences)}")

    band_counts = Counter(c[2] for c in cadences)
    for b, n in sorted(band_counts.items()):
        print(f"    {b}: {n} cadences")

    print(f"\n  PNG estimate per cadence (n_cadences={len(cadences)}):")
    print(f"  {'Step':>8}  {'Overlap':>10}  {'PNG/cadence':>14}  {'Total PNG':>12}")
    print(f"  {'-'*52}")
    for step in [VIZ_CHANS, VIZ_CHANS//2, VIZ_CHANS//4, 1024]:
        if step < 1:
            continue
        n_windows   = max(0, (n_freq - VIZ_CHANS) // step + 1)
        overlap_pct = max(0, (1 - step / VIZ_CHANS) * 100)
        total       = n_windows * len(cadences)
        print(f"  {step:>8}  {overlap_pct:>9.0f}%  {n_windows:>14,}  {total:>12,}")

    print(f"\n  Use --max-windows-per-cadence to cap the number of PNGs per cadence.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert SRT .h5 files to candidate PNGs for the srtad pipeline"
    )
    parser.add_argument("--scan", "-s", nargs="+", required=True,
                        help="Directories to scan for .h5 files (recursive)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for PNGs (default: real_png_dir from config)")
    parser.add_argument("--step", type=int, default=VIZ_CHANS,
                        help=f"Stride between windows in channels (default: {VIZ_CHANS} = no overlap)")
    parser.add_argument("--max-cadences", type=int, default=None,
                        help="Maximum number of cadences to process")
    parser.add_argument("--max-windows-per-cadence", type=int, default=None,
                        help="Maximum number of PNG windows per cadence")
    parser.add_argument("--skip-cadences", type=int, default=0,
                        help="Skip first N cadences (for resuming interrupted runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print how many PNGs would be generated without writing anything")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.scan)
        return

    output_dir = Path(args.output) if args.output else Path(paths["real_png_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir : {output_dir}")
    print(f"VIZ_CHANS  : {VIZ_CHANS}")
    print(f"Step       : {args.step}")

    all_files: List[Path] = []
    for d in args.scan:
        all_files.extend(Path(d).rglob("*.h5"))
    print(f".h5 files  : {len(all_files)}")

    cadences = group_into_cadences(all_files)
    print(f"Cadences   : {len(cadences)}")

    band_counts = Counter(c[2] for c in cadences)
    for b, n in sorted(band_counts.items()):
        print(f"  {b}: {n} cadences")

    if args.max_cadences:
        cadences = cadences[:args.max_cadences]
        print(f"Processing first {len(cadences)} cadences")

    cadences = interleave_by_band(cadences)

    if args.skip_cadences > 0:
        print(f"\nSkipping first {args.skip_cadences} cadences (resume mode)")
        cadences = cadences[args.skip_cadences:]

    candidate_idx = 0

    print(f"\nProcessing {len(cadences)} cadences...")

    total_saved = 0
    errors = []

    iterator = tqdm(cadences, desc="Processing cadences") if tqdm is not None else cadences

    for target, file_paths, band in iterator:
        try:
            n = process_cadence(target, file_paths, output_dir,
                              start_idx=candidate_idx,
                              step=args.step,
                              max_windows=args.max_windows_per_cadence)
            candidate_idx += n
            total_saved += n
        except Exception as e:
            errors.append(f"{target} ({band}): {e}")
            if tqdm is None:
                print(f"[WARN] Skipping {target} ({band}): {e}")
            continue

    if errors:
        print("\n[WARN] Errors during processing:")
        for err in errors:
            print(f"  {err}")

    print(f"\n[OK] Saved {total_saved} new PNGs to {output_dir}")
    final_count = len(list(output_dir.glob("*.png")))
    print(f"Total PNGs in directory: {final_count:,}")


if __name__ == "__main__":
    main()