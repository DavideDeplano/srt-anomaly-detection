"""
Visualization module.
Handles result display, updates, and dashboard interactions.
"""

from pathlib import Path
from sklearn.neighbors import KernelDensity
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

from src.srtad.config import paths
from src.srtad.core.candidate import Candidate

class Visualizer:
    """
    Handles generation of diagnostic plots and candidate visualizations.
    """

    def __init__(self) -> None:
        self._output_dir = Path(paths["results"]) / "figures"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("srtad.visualizer")

    def plot_umap_embedding(
        self, 
        Z: np.ndarray, 
        labels: np.ndarray, 
        title: str,           
        filename: str,         
        on_cat: int,         
        off_cat: int,         
        kde_on=KernelDensity,          
        kde_off=KernelDensity,  
    ) -> None:
        if Z.shape[1] != 2:
            self._logger.warning("Cannot plot UMAP embedding: dimensions != 2")
            return

        # Determine plot limits with padding
        x_min, x_max = Z[:, 0].min(), Z[:, 0].max()
        y_min, y_max = Z[:, 1].min(), Z[:, 1].max()
        
        pad_x = (x_max - x_min) * 0.1
        pad_y = (y_max - y_min) * 0.1
        
        # Bins definition
        bins_x = np.linspace(x_min - pad_x, x_max + pad_x, 100)
        bins_y = np.linspace(y_min - pad_y, y_max + pad_y, 100)

        # Grid definition
        fig = plt.figure(figsize=(10, 10))
        grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)

        ax_main = fig.add_subplot(grid[1:, :-1])
        ax_top = fig.add_subplot(grid[0, :-1], sharex=ax_main)
        ax_right = fig.add_subplot(grid[1:, -1], sharey=ax_main)

        # KDE CONTOURS
        if kde_on is not None and kde_off is not None:
            # Create grid for KDE evaluation
            x_range = np.linspace(bins_x[0], bins_x[-1], 100)
            y_range = np.linspace(bins_y[0], bins_y[-1], 100)
            xx, yy = np.meshgrid(x_range, y_range)
            grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
            
            # Calculate densities
            log_den_on = kde_on.score_samples(grid_points)
            log_den_off = kde_off.score_samples(grid_points)
            den_on = np.exp(log_den_on).reshape(xx.shape)
            den_off = np.exp(log_den_off).reshape(xx.shape)
            
            # Plot contour LINES (not filled)
            ax_main.contour(xx, yy, den_on, levels=5, colors='lightgreen', alpha=0.6, linewidths=0.8)
            ax_main.contour(xx, yy, den_off, levels=5, colors='lightcoral', alpha=0.6, linewidths=0.8)

        # Masks for categories
        mask_on = (labels == on_cat)
        mask_off = (labels == off_cat)
        mask_bg = (~mask_on) & (~mask_off)

        # 1. SCATTER PLOT (Central) 
        if np.any(mask_on):
            ax_main.scatter(
                Z[mask_on, 0], Z[mask_on, 1], 
                s=3, c='lime', alpha=0.8, label=f'Only ON (Cat {on_cat})'
            )

        if np.any(mask_off):
            ax_main.scatter(
                Z[mask_off, 0], Z[mask_off, 1], 
                s=3, c='red', alpha=0.8, label=f'Only OFF (Cat {off_cat})'
            )

        ax_main.scatter(
            Z[mask_bg, 0], Z[mask_bg, 1], 
            s=1, c='black', alpha=0.5, label='Mixed / Noise'
        )

        ax_main.grid(True, alpha=0.2)
        ax_main.set_xlabel("UMAP X")
        ax_main.set_ylabel("UMAP Y")
        ax_main.legend(loc='upper right', markerscale=3.0)
        
        # Set limits explicitly
        ax_main.set_xlim(bins_x[0], bins_x[-1])
        ax_main.set_ylim(bins_y[0], bins_y[-1])

        # 2. TOP MARGINAL (Histogram X) 
        ax_top.hist(Z[mask_bg, 0], bins=bins_x, color='black', alpha=0.3, density=True)
        if np.any(mask_off):
            ax_top.hist(Z[mask_off, 0], bins=bins_x, color='red', alpha=0.6, density=True)
        if np.any(mask_on):
            ax_top.hist(Z[mask_on, 0], bins=bins_x, color='lime', alpha=0.6, density=True)
        
        ax_top.axis('off')
        ax_top.set_title(title, fontsize=14, pad=20)

        # 3. RIGHT MARGINAL (Histogram Y) 
        ax_right.hist(Z[mask_bg, 1], bins=bins_y, orientation='horizontal', color='black', alpha=0.3, density=True)
        if np.any(mask_off):
            ax_right.hist(Z[mask_off, 1], bins=bins_y, orientation='horizontal', color='red', alpha=0.6, density=True)
        if np.any(mask_on):
            ax_right.hist(Z[mask_on, 1], bins=bins_y, orientation='horizontal', color='lime', alpha=0.6, density=True)
        
        ax_right.axis('off')

        # Saving
        out_path = self._output_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        self._logger.info(f"Saved UMAP plot with KDE contours to {out_path}")
        
    def plot_density_histogram(
        self, 
        densities: np.ndarray, 
        threshold: float = 0.0618,
        filename: str = "density_histogram.png"
    ) -> None:
        if densities is None or densities.size == 0:
            return

        densities = densities[densities > 0]

        if densities.size == 0:
            return

        plt.figure(figsize=(10, 6))
        plt.hist(densities, bins=100, color='steelblue', log=True, edgecolor='none')
        
        plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold {threshold}')
        
        plt.title("Distribution of 'Only-ON' Probabilities")
        plt.xlabel("Probability (Density)")
        plt.ylabel("Counts (Log Scale)")
        plt.legend()
        
        out_path = self._output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        self._logger.info(f"Saved density histogram to {out_path}")

    def plot_frequency_histogram_by_band(
        self,
        candidates: List[Candidate],
        filename: str = "frequency_histogram_by_band.png",
        bins: int = 200,
    ) -> None:
        """
        Histogram of candidate frequencies per band (log Y).
        """
        if not candidates:
            return

        bands = ["C", "K"]
        band_to_freq_mhz: Dict[str, List[float]] = {b: [] for b in bands}

        for c in candidates:
            b = getattr(c, "band", None)
            if b not in band_to_freq_mhz:
                continue
            band_to_freq_mhz[b].append(float(c.frequency_hz) / 1e6)

        fig, axes = plt.subplots(1, len(bands), figsize=(14, 4), sharey=False)
        if len(bands) == 1:
            axes = [axes]

        for ax, b in zip(axes, bands):
            freqs = np.array(band_to_freq_mhz[b], dtype=float)
            if freqs.size == 0:
                ax.set_title(f"{b} Band (no data)")
                ax.set_xlabel("Frequency [MHz]")
                ax.set_ylabel("Counts")
                ax.set_yscale("log")
                continue

            ax.hist(freqs, bins=bins)
            ax.set_title(f"{b} Band")
            ax.set_xlabel("Frequency [MHz]")
            ax.set_ylabel("Counts")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.2)

        out_path = self._output_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        self._logger.info(f"Saved frequency histogram by band to {out_path}")

    def plot_frequency_score_histogram_by_band(
        self,
        candidates: List[Candidate],
        filename: str = "frequency_score_histogram_by_band.png",
        bins: int = 100,
    ) -> None:
        """
        Paper-like Figure 6 (adapted):
        Histogram of frequency_score per band (log Y).
        """
        if not candidates:
            return

        bands = ["C", "K"]
        band_to_scores: Dict[str, List[float]] = {b: [] for b in bands}

        for c in candidates:
            b = getattr(c, "band", None)
            s = getattr(c, "frequency_score", None)
            if b not in band_to_scores:
                continue
            if s is None:
                continue
            band_to_scores[b].append(float(s))

        fig, axes = plt.subplots(1, len(bands), figsize=(14, 4), sharey=False)
        if len(bands) == 1:
            axes = [axes]

        for ax, b in zip(axes, bands):
            scores = np.array(band_to_scores[b], dtype=float)
            if scores.size == 0:
                ax.set_title(f"{b} Band (no scores)")
                ax.set_xlabel("Frequency score")
                ax.set_ylabel("Counts")
                ax.set_yscale("log")
                continue

            scores = np.clip(scores, 0.0, 1.0)

            ax.hist(scores, bins=bins, range=(0.0, 1.0))
            ax.set_title(f"{b} Band")
            ax.set_xlabel("Frequency score")
            ax.set_ylabel("Counts")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.2)

        out_path = self._output_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        self._logger.info(f"Saved frequency score histogram by band to {out_path}")

    def plot_similarity_umap(
        self,
        Z_all: np.ndarray,
        Z_good: np.ndarray,
        Z_bad: np.ndarray,
        s_good: float,
        s_bad: float,
        filename: str = "similarity_umap.png",
        on_idx: tuple[int, int, int] = (0, 2, 4),
        off_idx: tuple[int, int, int] = (1, 3, 5),
    ) -> None:
        """
        Paper-like Figure 7 for similarity filter:
        - Background: all embedded panels (black cloud)
        - Overlay: one "good" candidate and one "bad" candidate
        annotated with letters A (ON panels) and B (OFF panels)
        - Legend reports S_sim for both examples

        Inputs
        ------
        Z_all  : (N, 2)   UMAP embedding of ALL panels from all candidates
        Z_good : (6, 2)   UMAP embedding of the 6 panels for a high-similarity candidate
        Z_bad  : (6, 2)   UMAP embedding of the 6 panels for a low-similarity candidate
        """
        if Z_all.ndim != 2 or Z_all.shape[1] != 2:
            self._logger.warning("Z_all must be (N,2).")
            return
        if Z_good.ndim != 2 or Z_good.shape != (6, 2):
            self._logger.warning("Z_good must be (6,2).")
            return
        if Z_bad.ndim != 2 or Z_bad.shape != (6, 2):
            self._logger.warning("Z_bad must be (6,2).")
            return

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # Background cloud (all panels)
        ax.scatter(Z_all[:,0], Z_all[:,1], s=2, c="black", alpha=0.5, linewidths=0)

        # Helper to draw A/B letters for one candidate
        def _annotate_candidate(Z6: np.ndarray, color: str) -> None:
            for i in range(6):
                letter = "A" if i in on_idx else "B"
                ax.text(
                    Z6[i, 0],
                    Z6[i, 1],
                    letter,
                    color=color,
                    fontsize=14,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

        # Overlay letters (paper-style)
        _annotate_candidate(Z_good, color="green")
        _annotate_candidate(Z_bad, color="red")

        ax.set_xlabel("X UMAP")
        ax.set_ylabel("Y UMAP")
        ax.grid(True, alpha=0.25)

        # Legend proxies: A/B for good/bad with S_sim values
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color="green", linestyle="None",
                label=f"High similarity (S_sim = {s_good:.2f})"),
            Line2D([0], [0], marker="o", color="red", linestyle="None",
                label=f"Low similarity (S_sim = {s_bad:.2f})"),
        ]

        ax.legend(handles=legend_handles, loc="upper right", frameon=False)

        out_path = self._output_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        self._logger.info(f"Saved similarity UMAP plot to {out_path}")
