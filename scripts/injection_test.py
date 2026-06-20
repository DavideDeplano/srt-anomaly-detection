"""
injection_test.py

Positive-control test for the srtad Density Filter.

Takes one real SRT cadence, injects a drifting narrowband signal ONLY into
the ON observations (with a drift that progresses coherently through the
cadence timeline, as a real sky signal would), at several amplitude levels,
and renders the PNGs with the current production pipeline.

The resulting PNGs (one per amplitude) are written to a dedicated folder.
Run the Density Filter (option 3) on that folder: candidates with a
sufficiently strong injected signal must come out as category 42 and pass.
The lowest passing amplitude is the empirical sensitivity of the pipeline.

Run from the project root on Colab:
    PYTHONPATH=".:src" python scripts/injection_test.py
"""

import warnings
from pathlib import Path

import numpy as np
from blimpy import Waterfall

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.h5_to_candidates import (
    group_into_cadences,
    _preprocess_cadence,
    _save_png,
    VIZ_CHANS,
)

# ── settings ──────────────────────────────────────────────────────────────────

TARGET_NAME = "TIC371234684"                 # cadence to use
SCAN_DIR    = "/content/nvme_esterno"
OUTPUT_DIR  = "/content/davide/srt-anomaly-detection/data/injection_test_nodrift"

# Amplitudes of the injected line, in units of the local noise std
SNR_LEVELS  = [5, 10, 20, 50, 100, 200]

# Window widths to test, in raw channels. 4096 is the production window;
# 640 gives 8 channels per loader column, close to the synthetic training
# scale (frames are 16x80 at df=2.79 Hz, i.e. 224 Hz wide).
WINDOW_CHANS = [4096, 640]

DRIFT_HZ_S  = 0.1          # drift rate of the injected signal [Hz/s]
LINE_WIDTH  = 2            # line width in raw channels
OBS_GAP_S   = 319.0        # approximate start-to-start spacing between observations


def pick_quiet_window(cadence: np.ndarray, width: int) -> int:
    """
    Pick the start of the window (of the given width) with the flattest
    content: minimizes the max-over-time peak so the injection lands on
    clean noise.
    """
    n_freq = cadence.shape[2]
    best_start, best_peak = 0, np.inf
    prof = cadence.max(axis=(0, 1))  # max over panels and time, per channel
    med = np.median(prof)
    for start in range(0, n_freq - width + 1, width):
        peak = prof[start:start + width].max() / med
        if peak < best_peak:
            best_peak, best_start = peak, start
    return best_start


def main():
    files = [f for f in Path(SCAN_DIR).rglob("*.h5") if TARGET_NAME in f.name]
    cadences = group_into_cadences(files)
    if not cadences:
        print(f"[ERROR] No cadence found for {TARGET_NAME}")
        return
    target, fps, band = cadences[0]

    print(f"Loading cadence {target}...")
    obs, freq_axis = [], None
    dt_s = None
    for fp in fps:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf = Waterfall(str(fp))
        d = wf.data.squeeze().astype(np.float32)
        obs.append(d)
        if freq_axis is None:
            fch1 = wf.header.get("fch1", 0.0)
            foff = float(wf.header.get("foff", 0.0))
            dt_s = float(wf.header.get("tsamp", 18.25))
            freq_axis = fch1 + foff * np.arange(d.shape[1])

    min_t = min(d.shape[0] for d in obs)
    obs = [d[:min_t] for d in obs]
    cadence = np.stack(obs, axis=0)          # (6, T, n_freq)
    n_panels, T, n_freq = cadence.shape
    foff_hz = (freq_axis[1] - freq_axis[0]) * 1e6   # signed channel width [Hz]

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for width in WINDOW_CHANS:
        start = pick_quiet_window(cadence, width)
        end = start + width
        center_chan = start + width // 2
        freq_mhz = float(freq_axis[center_chan])
        print(f"\n[width {width}] quiet window: channels {start}-{end}, "
              f"centre {freq_mhz:.6f} MHz")

        window = cadence[:, :, start:end].astype(np.float64)
        noise_std = float(np.std(window))

        # Channel drift per time sample (sign-corrected: fixed sky direction)
        chan_per_s = DRIFT_HZ_S / abs(foff_hz)

        for snr in SNR_LEVELS:
            injected = window.copy()
            amp = snr * noise_std

            total_drift_chan = chan_per_s * (5 * OBS_GAP_S + T * dt_s)
            line_start = width // 2 - total_drift_chan / 2

            for p in (0, 2, 4):                      # ON panels only
                # t0 advances with the observation start time: the injected
                # signal drifts coherently through the whole cadence timeline,
                # as a real sky signal would between consecutive observations.
                # With DRIFT_HZ_S = 0 the line is perfectly vertical and lands
                # at the same frequency in every ON panel.
                t0 = p * OBS_GAP_S
                for t in range(T):
                    pos = line_start + chan_per_s * (t0 + t * dt_s)
                    c = int(round(pos))
                    if 0 <= c < width - LINE_WIDTH:
                        injected[p, t, c:c + LINE_WIDTH] += amp

            viz = injected
            if foff_hz < 0:
                viz = viz[:, :, ::-1]
            processed = _preprocess_cadence(viz)

            fname = f"INJ_W{width}_SNR{snr:04d}_{target}_{freq_mhz:.6f}MHz.png"
            _save_png(processed, f"INJ-W{width}-SNR{snr}", freq_mhz, out_dir,
                      filename=fname)
            print(f"  [OK] width {width}, SNR {snr:>4}: {fname}")

    print(f"\n[DONE] {len(SNR_LEVELS)} PNGs in {out_dir}")
    print("Run the Density Filter (option 3) on this folder.")
    print("Expected: high-SNR candidates -> category 42 and pass; the lowest")
    print("passing SNR is the pipeline sensitivity.")


if __name__ == "__main__":
    main()