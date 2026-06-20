"""
Generate PNG for EXACT frequency from .h5 files - one per cadence
"""

from pathlib import Path
from blimpy import Waterfall
import warnings
import numpy as np
from scripts.h5_to_candidates import (
    group_into_cadences,
    _preprocess_cadence,
    _save_png,
    VIZ_CHANS
)


def generate_png_for_frequency(target_freq_mhz: float,
                               output_dir: str,
                               scan_dir: str = "/content/nvme_esterno"):
    """
    Generate PNG for ALL cadences at EXACT frequency.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_files = list(Path(scan_dir).rglob("*.h5"))
    cadences = group_into_cadences(all_files)

    print(f"Searching {len(cadences)} cadences for frequency {target_freq_mhz:.6f} MHz...\n")

    saved = 0
    processed_targets = set()

    for target_name, file_paths, band in cadences:
        # Duplicate .h5 copies in different directories produce duplicate
        # cadences for the same target: process each target once
        if target_name in processed_targets:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wf = Waterfall(str(file_paths[0]), load_data=False)
            fch1 = float(wf.header.get("fch1", 0.0))
            foff = float(wf.header.get("foff", 0.0))
            nchans = int(wf.header.get("nchans", 0))
            freq_min = min(fch1, fch1 + foff * nchans)
            freq_max = max(fch1, fch1 + foff * nchans)

            if not (freq_min <= target_freq_mhz <= freq_max):
                continue

            print(f"Loading {target_name} ({freq_min:.2f}-{freq_max:.2f} MHz)...")

            obs_data = []
            freq_axis = None
            foff_data = 0.0

            for fp in file_paths:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wf = Waterfall(str(fp))
                data = wf.data.squeeze()

                if obs_data and data.shape[1] != obs_data[0].shape[1]:
                    raise ValueError(
                        f"Channel count mismatch: {data.shape} vs {obs_data[0].shape}"
                    )

                obs_data.append(data.astype(np.float32))

                if freq_axis is None:
                    header = wf.header
                    fch1_data = header.get("fch1", 0.0)
                    foff_data = float(header.get("foff", 0.0))
                    nchans_data = data.shape[1]
                    freq_axis = fch1_data + foff_data * np.arange(nchans_data)

            # Some observations have one extra time integration (e.g. 17 vs 16):
            # trim all panels to the common time length before stacking
            min_t = min(d.shape[0] for d in obs_data)
            obs_data = [d[:min_t] for d in obs_data]

            cadence_array = np.stack(obs_data, axis=0)
            n_freq = cadence_array.shape[2]

            # Find EXACT channel closest to target frequency
            center_chan = int(np.argmin(np.abs(freq_axis - target_freq_mhz)))

            start = center_chan - VIZ_CHANS // 2
            end = start + VIZ_CHANS

            if start < 0:
                start = 0
                end = VIZ_CHANS
            if end > n_freq:
                end = n_freq
                start = n_freq - VIZ_CHANS

            # Frequency of the actual window centre channel
            window_center_chan = start + VIZ_CHANS // 2
            center_freq_mhz = float(freq_axis[window_center_chan])

            print(f"  Window: channels {start}-{end}, centre {center_freq_mhz:.6f} MHz")

            viz_window = cadence_array[:, :, start:end]

            # If foff < 0, channel index runs from high to low frequency:
            # flip so that the x axis increases with frequency
            if foff_data < 0:
                viz_window = viz_window[:, :, ::-1]

            processed = _preprocess_cadence(viz_window)

            # Use _save_png with custom filename
            fname = f"{target_name}_{target_freq_mhz:.6f}MHz.png"
            png_path = _save_png(processed, target_name, center_freq_mhz,
                                 output_path, filename=fname)

            saved += 1
            processed_targets.add(target_name)
            print(f"  [OK] Saved: {fname}")

        except Exception as e:
            print(f"  Skipping {target_name}: {e}")
            continue

    print(f"\n[DONE] Saved {saved} PNGs to {output_path}")


if __name__ == "__main__":
    for freq in [4859.977720, 4950.025520]:
        generate_png_for_frequency(
            target_freq_mhz=freq,
            output_dir="/content/davide/srt-anomaly-detection/data/custom_candidates",
            scan_dir="/content/nvme_esterno"
        )