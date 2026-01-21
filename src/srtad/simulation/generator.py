"""
Synthetic cadence generator (Setigen-based).

This module generates simulated SETI-like cadences composed of 6 panels
(ON/OFF slots). Each cadence is generated for all binary ON/OFF masks
(patterns), producing a total of:

    total_cadences = n_panel_sets * mask_combinations

Outputs
-------
For each (cadence_id, pattern_id):
- A tensor saved as .npy with shape (6, tchans, fchans)
- Optionally, a diagnostic PNG waterfall plot (random subsampling)

A CSV log file is also produced to store slot-level metadata for traceability
(e.g., ON/OFF flag, frequency start index, drift rate, profile type, seed).
"""

from astropy import units as u
import setigen as stg
import numpy as np
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from ..config import simulation as sim_cfg, paths, random_seed
import logging
from scipy.ndimage import zoom
import inspect

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger("srtad.simulation")

# Slot type labels for the 6-panel cadence
SLOT_TYPES = ["A", "B", "A", "C", "A", "D"]

class SimulationGenerator:
    """
    Generate synthetic SETI observation cadences.

    Each cadence is made of 6 panels (slots). For each cadence_id, the generator
    produces one tensor per binary ON/OFF mask (pattern_id).

    A single "panel set" corresponds to:
    - a fixed background noise realization (copied into each slot frame)
    - fixed signal parameters sampled once per pattern (shared across ON slots)

    The output tensor has shape:
        (panels_per_cadence, tchans, fchans) == (6, tchans, fchans)
    """
    def __init__(self):
        self.sim = sim_cfg
        self.data_dir = Path(paths["data"])
        self.output_dir = self.data_dir / sim_cfg["output_cadences_dir"]
        self.plots_dir = self.data_dir / sim_cfg["output_waterfall_plots_dir"]
        self.seed = int(random_seed)

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Guard: if any output already exists, skip regeneration
        if any(self.output_dir.glob("cadence_*_pattern*.npy")):
            logger.info(f"Existing data found in {self.output_dir}. Skipping.")
            return

        n_panel_sets = int(self.sim["n_panel_sets"])                # number of cadence sets
        panels_per_cadence = int(self.sim["panels_per_cadence"])    # must be 6 for this pipeline
        mask_combinations = int(self.sim["mask_combinations"])      # 2^6 = 64
        total_cadences = n_panel_sets * mask_combinations

        print(f"Generating {total_cadences} cadences...")

        tchans = int(self.sim["tchans"])                            # time bins per panel
        fchans = int(self.sim["fchans"])                            # frequency bins per panel
        
        amplitude_factor_range = tuple(self.sim["amplitude_factor"])
        drift_rate_range = tuple(self.sim["drift_rate_hz_s"])
        width_hz_range = tuple(self.sim["width_hz"])
        
        df_hz = float(self.sim["df_hz"])                            # frequency resolution (Hz/bin)
        dt_s = float(self.sim["dt_s"])                              # time resolution (s/bin)
        fch1_mhz = float(self.sim["fch1_mhz"])                      # start frequency (MHz)


        # Load real background noise
        noise_path = self.data_dir / "noise" / "background_noise.npy"
        background = np.load(noise_path).astype(np.float64)

        # Enforce exact expected shape for background
        if background.shape != (tchans, fchans):
            raise ValueError(f"background_noise.npy shape {background.shape} != ({tchans},{fchans})")

        # Sanity check: cadence slot definition must match configured panel count  
        if panels_per_cadence != len(SLOT_TYPES):
            raise ValueError("Panel count mismatch.")
        
        # Discover all available frequency profile functions in this setigen install 
        F_PROFILE_FNS = []
        for name in dir(stg):
            if name.endswith("_f_profile"):
                fn = getattr(stg, name, None)
                if callable(fn):
                    F_PROFILE_FNS.append((name, fn))

        if not F_PROFILE_FNS:
            raise RuntimeError("No *_f_profile functions found in setigen.")
        
        F_PROFILE_FNS.sort(key=lambda x: x[0])

        # Discover time profile functions that accept ONLY a "level" parameter
        T_PROFILE_FNS = []
        for name in dir(stg):
            if not name.endswith("_t_profile"):
                continue

            fn = getattr(stg, name, None)
            if not callable(fn):
                continue

            try:
                sig = inspect.signature(fn)
                params = list(sig.parameters.values())

                # Accept ONLY functions with exactly one parameter named "level"
                if len(params) == 1 and params[0].name == "level":
                    T_PROFILE_FNS.append((name, fn))

            except Exception:
                continue

        # Fallback: constant time profile if no suitable time builders found
        if not T_PROFILE_FNS:
            T_PROFILE_FNS = [("constant_t_profile", stg.constant_t_profile)]

        T_PROFILE_FNS.sort(key=lambda x: x[0])
        
        # CSV log file: slot-level metadata for traceability and debugging
        log_path = self.output_dir / "cadences_log.csv"
        fieldnames = [
            "cadence_id", "pattern_id", "slot", "slot_type", "on", 
            "amplitude_factor", "drift_rate_hz_s", "width_hz",
            "f_start_mhz", "f_start_idx", "signal_level", "signal_profile", 
            "tchans", "fchans", "df_hz", "dt_s", "fch1_mhz", "random_seed",
        ]

        with log_path.open("w", newline="") as log_f:
            writer = csv.DictWriter(log_f, fieldnames=fieldnames)
            writer.writeheader()

            iterator = tqdm(range(n_panel_sets), desc="Sets") if tqdm else range(n_panel_sets)

            # Main generation loop: iterate over independent cadence sets
            for cadence_id in iterator:
                batch_seed = self.seed + cadence_id
                rng = np.random.default_rng(batch_seed)
                
                # 1) Build base frames for each slot using the same real background
                base_panels = []
                for slot in range(panels_per_cadence):
                    frame = stg.Frame(
                        fchans=fchans, tchans=tchans,
                        df=df_hz * u.Hz,
                        dt=dt_s * u.s,
                        fch1=fch1_mhz * u.MHz,
                    )
                    frame.data = background.copy()
                    base_panels.append(frame)


                # 2) For each ON/OFF mask (pattern_id), generate one cadence tensor
                for pattern_id in range(mask_combinations):
                    # Extract ON/OFF bits from pattern_id, one bit per slot
                    # bits[k] == 1 means "slot k is ON"
                    bits = [(pattern_id >> k) & 1 for k in range(panels_per_cadence)]
                    panels = []

                    # 2.1) Sample parameters ONCE per (cadence_id, pattern_id)
                    max_width_hz = float(width_hz_range[1])
                    half_width_bins = int(np.ceil((max_width_hz / df_hz) / 2.0))
                    if 2 * half_width_bins >= fchans:
                        half_width_bins = 0
                    start_f_idx = int(rng.integers(half_width_bins, fchans - half_width_bins))

                    amp_mult = drift = width = 0.0
                    if any(bits):
                        amp_mult = float(rng.uniform(*amplitude_factor_range))
                        drift = float(rng.uniform(*drift_rate_range))
                        width = float(rng.uniform(*width_hz_range))

                     # 2.2) Build the 6 panels for this pattern
                    for slot in range(panels_per_cadence):
                        frame = base_panels[slot].copy()
                        
                        # Signal start frequency is derived from the chosen index
                        f_idx = start_f_idx
                        f_start = frame.get_frequency(index=f_idx)
                        f_start_mhz = float(f_start)

                        on_flag = bool(bits[slot])

                        # Slot-level metadata defaults (OFF slots keep these at zero/none)
                        signal_level = 0.0
                        profile_name = "none" 

                        f_profile = None
                        t_profile = None

                        if on_flag:
                            # Signal level is defined relative to the local maximum noise
                            # This ties amplitude to the background scale
                            current_noise_max = np.max(frame.data)
                            signal_level = amp_mult * current_noise_max
                            
                            if signal_level > 0:
                                # Pick a frequency profile builder at random
                                profile_name, f_builder = F_PROFILE_FNS[
                                    int(rng.integers(0, len(F_PROFILE_FNS)))
                                ]

                                w = width * u.Hz

                                try:
                                    f_profile = f_builder(width=w)
                                except TypeError:
                                    try:
                                        f_profile = f_builder(w)
                                    except TypeError:
                                        f_profile = f_builder(w, w)

                                t_name, t_builder = T_PROFILE_FNS[int(rng.integers(0, len(T_PROFILE_FNS)))]
                                t_profile = t_builder(level=signal_level)

                                profile_name = f"{profile_name}|{t_name}"

                                # Add the signal with:
                                # - constant drift path
                                # - chosen time profile (amplitude over time)
                                # - chosen frequency profile (spectral shape)
                                # - constant bandpass profile
                                frame.add_signal(
                                    stg.constant_path(f_start=f_start, drift_rate=drift * u.Hz / u.s),
                                    t_profile,
                                    f_profile,
                                    stg.constant_bp_profile(level=1.0),
                                )
    
                        panels.append(frame.data.copy())

                        writer.writerow({
                            "cadence_id": f"cadence_{cadence_id:05d}",
                            "pattern_id": pattern_id,
                            "slot": slot,
                            "slot_type": SLOT_TYPES[slot],
                            "on": int(on_flag),
                            "amplitude_factor": amp_mult,
                            "drift_rate_hz_s": drift,
                            "width_hz": width,
                            "f_start_mhz": f_start_mhz,
                            "f_start_idx": f_idx,
                            "signal_level": float(signal_level),
                            "signal_profile": profile_name,
                            "tchans": tchans, "fchans": fchans,
                            "df_hz": df_hz, "dt_s": dt_s, "fch1_mhz": fch1_mhz,
                            "random_seed": batch_seed
                        })

                    # 2.3) Stack panels and save the cadence tensor as .npy
                    tensor = np.stack(panels, axis=0)
                    filename_base = f"cadence_{cadence_id:05d}_pattern{pattern_id:02d}"
                    np.save(self.output_dir / f"{filename_base}.npy", tensor)

                    # 2.4) Save a diagnostic PNG for a small random subset
                    # The PNG is a visualization artifact only; it is not used for training
                    if rng.random() < 0.05:
                         # Concatenate 6 panels vertically for a single waterfall image
                        full_waterfall = tensor.reshape(-1, fchans)

                        # Upscale for readability (nearest-neighbor via order=0)
                        scale = 8
                        large_waterfall = zoom(full_waterfall, (scale, scale), order=0)
                        png_path = self.plots_dir / f"{filename_base}.png"

                        plt.imsave(
                            png_path, 
                            large_waterfall, 
                            cmap='viridis', 
                            origin='lower',
                            dpi=300
                        )

                        plt.close('all')
                        
        logger.info(f"Dataset generated. NPY: {self.output_dir}, PNG: {self.plots_dir}")