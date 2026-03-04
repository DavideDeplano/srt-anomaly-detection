"""
UMAPReverseSearch 

Given a real candidate, project it into the trained UMAP space and
visualize where it falls relative to all other candidates.

Useful for:
- Understanding which cluster a candidate belongs to
- Finding morphologically similar candidates in the same UMAP region
- Investigating whether multiple candidates correspond to the same RFI type

Usage in Colab
--------------
    from scripts.umap_reverse_search import UMAPReverseSearch

    searcher = UMAPReverseSearch()
    searcher.load()

    # Project all real candidates and plot
    fig = searcher.plot()
    fig.show()

    # Highlight a specific candidate
    fig = searcher.plot(highlight_id="candidate_12_6924.92MHz_P0.810")
    fig.show()

    # Find neighbors in UMAP space
    neighbors = searcher.find_neighbors("candidate_12_6924.92MHz_P0.810", radius=2.0)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import joblib
import numpy as np

from src.srtad.config import filters, paths
from src.srtad.core.candidate import Candidate
from src.srtad.management.cross_correlation_extractor import CrossCorrelationExtractor
from src.srtad.core.dataset import Dataset


class UMAPReverseSearch:
    """
    Projects real SRT candidates into the trained UMAP density space
    and provides interactive visualization and nearest-neighbor search.

    The class reuses the UMAP model trained by DensityFilter, avoiding
    any additional training. It computes 15-dimensional cross-correlation
    features from each candidate's cadence and transforms them via the
    existing UMAP model.

    Attributes
    ----------
    _umap_model : fitted UMAP model loaded from disk
    _candidates : list of real Candidate objects successfully projected
    _Z          : (N, 2) UMAP coordinates for each candidate in _candidates
    _Z_val      : (M, 2) UMAP coordinates of the simulated validation set (background)
    _y_val      : (M,)   category labels of the simulated validation set

    Notes
    -----
    _candidates and _Z are always aligned: _candidates[i] corresponds to _Z[i].
    Candidates that fail feature extraction are skipped entirely (not zero-padded),
    so the plot only ever shows real projected data.
    """

    # Band colors used in the interactive plot
    _BAND_COLORS: Dict[str, str] = {
        "C": "#00e5ff",
        "K": "#bd93f9",
        "OUT_OF_BAND": "#888888",
    }

    def __init__(self) -> None:
        self._umap_path = Path(filters["density"]["umap_model_path"])
        self._kde_dir = Path(filters["density"]["kde_models_dir"])
        self._results_dir = Path(paths["results"])

        # Read category IDs from config — do not rely on magic numbers
        self._only_on_cat  = int(filters["density"]["only_on_category"])
        self._only_off_cat = int(filters["density"]["only_off_category"])

        self._extractor = CrossCorrelationExtractor()
        self._umap_model = None
        self._candidates: List[Candidate] = []
        self._Z: Optional[np.ndarray] = None
        self._Z_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None

    def load(self) -> "UMAPReverseSearch":
        """
        Load the UMAP model, real candidates, and optional validation background.

        Must be called before project(), plot(), or find_neighbors().

        Returns
        -------
        self : allows method chaining
        """
        print("[1/3] Loading UMAP model...")
        self._umap_model = self._load_umap()

        print("[2/3] Loading real candidates...")
        self._candidates = self._load_candidates()
        print(f"      → {len(self._candidates)} candidates with density_score > 0")

        print("[3/3] Loading simulated validation background...")
        self._Z_val, self._y_val = self._load_val_embedding()
        if self._Z_val is not None:
            print(f"      → {len(self._Z_val)} validation points loaded")
        else:
            print("      → validation background not available")

        return self

    def project(self) -> np.ndarray:
        """
        Project all loaded candidates into the UMAP space.

        Computes 15-dimensional CC features from each candidate's cadence
        and transforms them using the trained UMAP model.

        Candidates that are missing cadence data or fail feature extraction
        are skipped and logged — they are never zero-padded. After this call,
        self._candidates contains only the successfully projected candidates,
        aligned with self._Z row by row.

        Returns
        -------
        Z : np.ndarray of shape (N, 2)
            UMAP coordinates for each successfully projected candidate.

        Raises
        ------
        RuntimeError
            If no candidate can be projected (all cadences missing or broken).
        """
        self._check_loaded()

        print(f"Projecting {len(self._candidates)} candidates into UMAP space...")

        features: List[np.ndarray] = []
        valid_candidates: List[Candidate] = []
        n_skipped = 0

        for c in self._candidates:
            cadence = getattr(c, "cadence", None)

            if cadence is None:
                # Candidate was pickled without cadence data — skip entirely
                n_skipped += 1
                continue

            try:
                feat = self._extractor.extract_features(cadence)
                features.append(feat)
                valid_candidates.append(c)
            except Exception as exc:
                # Log the specific failure so the caller can investigate
                print(f"[WARN] Skipping {c.id}: {type(exc).__name__}: {exc}")
                n_skipped += 1

        if n_skipped > 0:
            print(
                f"[WARN] {n_skipped}/{len(self._candidates)} candidates skipped "
                f"(missing cadence or extraction error). "
                f"Ensure the pickle was saved with cadence data included."
            )

        if not features:
            raise RuntimeError(
                "No valid candidates to project — all cadences are missing or "
                "failed extraction. Cannot build UMAP plot."
            )

        # Keep _candidates aligned with _Z: only successfully projected ones
        self._candidates = valid_candidates

        X = np.array(features, dtype=np.float64)
        self._Z = self._umap_model.transform(X)
        print(
            f"Projection complete: {len(self._candidates)} candidates, "
            f"shape {self._Z.shape}"
        )
        return self._Z

    def plot(
        self,
        highlight_id: Optional[str] = None,
        max_bg_points: int = 30_000,
        save_html: bool = True,
    ) -> go.Figure:
        """
        Generate an interactive Plotly scatter plot in UMAP space.

        The plot shows:
        - Simulated validation set as background (mixed/noise in gray,
          only-ON in green, only-OFF in red)
        - Real candidates colored by observing band (C=cyan, K=purple)
        - Hover tooltips with ID, frequency, drift, band, and all scores
        - Optional star marker to highlight a specific candidate

        Parameters
        ----------
        highlight_id : str, optional
            ID of the candidate to highlight with a yellow star marker.
        max_bg_points : int
            Maximum number of background simulation points to render
            (subsampled for performance, default 30 000).
        save_html : bool
            If True, saves the interactive plot to results/umap_reverse_search.html.

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """
        # Project candidates if not done yet
        if self._Z is None:
            self.project()

        fig = go.Figure()

        # Add simulated validation set as background
        if self._Z_val is not None and self._y_val is not None:
            fig = self._add_background(fig, max_bg_points)

        # Add real candidates colored by band
        fig = self._add_candidates(fig, highlight_id)

        fig.update_layout(
            title=dict(
                text="UMAP Reverse Search — Real SRT Candidates",
                font=dict(size=16, color="#ccd6f6"),
            ),
            plot_bgcolor="#0b0c10",
            paper_bgcolor="#080a0f",
            font=dict(color="#8892b0", family="monospace"),
            legend=dict(
                bgcolor="rgba(15,17,23,0.9)",
                bordercolor="#1c2035",
                borderwidth=1,
                font=dict(size=11),
            ),
            xaxis=dict(title="UMAP X", gridcolor="#1c2035", zerolinecolor="#1c2035"),
            yaxis=dict(title="UMAP Y", gridcolor="#1c2035", zerolinecolor="#1c2035"),
            width=950,
            height=780,
            hovermode="closest",
        )

        if save_html:
            out_path = self._results_dir / "umap_reverse_search.html"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out_path), full_html=True, include_plotlyjs=True)
            print(f"[OK] Interactive plot saved to: {out_path}")

        return fig

    def find_neighbors(
        self,
        candidate_id: str,
        radius: float = 2.0,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Find the nearest candidates to a target in the UMAP space.

        Searches within a Euclidean radius and returns the closest matches,
        ordered by ascending distance. Useful for identifying candidates
        that share morphological characteristics with the target.

        Note: UMAP does not preserve Euclidean distances reliably. Results
        are meaningful for visual exploration but should not be treated as
        physically robust similarity measures.

        Parameters
        ----------
        candidate_id : str
            ID of the reference candidate.
        radius : float
            Search radius in UMAP coordinates (default 2.0).
        top_k : int
            Maximum number of neighbors to return (default 10).

        Returns
        -------
        neighbors : list of dict
            Each dict contains: id, distance, frequency_mhz, band,
            density_score, frequency_score, similarity_score, umap_x, umap_y.
        """
        if self._Z is None:
            self.project()

        # Locate the target candidate by ID
        target_idx = next(
            (i for i, c in enumerate(self._candidates) if c.id == candidate_id),
            None,
        )
        if target_idx is None:
            print(f"[WARN] Candidate '{candidate_id}' not found.")
            return []

        z_target = self._Z[target_idx]

        # Compute pairwise Euclidean distances from the target
        dists = np.linalg.norm(self._Z - z_target, axis=1)

        neighbors = []
        for i, (c, d) in enumerate(zip(self._candidates, dists)):
            # Skip the target itself and candidates outside the search radius
            if i == target_idx or d > radius:
                continue
            neighbors.append({
                "id": c.id,
                "distance": float(d),
                "frequency_mhz": c.frequency_hz / 1e6,
                "band": getattr(c, "band", "?"),
                "density_score": c.density_score,
                "frequency_score": getattr(c, "frequency_score", None),
                "similarity_score": getattr(c, "similarity_score", None),
                "umap_x": float(self._Z[i, 0]),
                "umap_y": float(self._Z[i, 1]),
            })

        # Sort by distance and cap at top_k
        neighbors.sort(key=lambda x: x["distance"])
        neighbors = neighbors[:top_k]

        print(f"\nReference: {candidate_id}")
        print(f"UMAP position: ({z_target[0]:.3f}, {z_target[1]:.3f})")
        print(f"Neighbors within radius {radius}: {len(neighbors)}")
        print("-" * 60)
        for n in neighbors:
            print(f"  d={n['distance']:.3f} | {n['id']}")
            print(f"           freq={n['frequency_mhz']:.3f} MHz | band={n['band']}")
            if n["density_score"] is not None:
                print(f"           density={n['density_score']:.4e}")

        return neighbors

    def _check_loaded(self) -> None:
        """Raise RuntimeError if load() has not been called yet."""
        if self._umap_model is None or not self._candidates:
            raise RuntimeError("Call load() before using this method.")

    def _load_umap(self):
        """Load the trained UMAP model from disk."""
        if not self._umap_path.exists():
            raise FileNotFoundError(
                f"UMAP model not found: {self._umap_path}\n"
                "Train the density filter first (option 2 in main.py)."
            )
        return joblib.load(self._umap_path)

    def _load_candidates(self) -> List[Candidate]:
        """
        Load real candidates directly from the configured real_png_dir.

        """
        real_dir = Path(paths["real_png_dir"])
        ds = Dataset(png_dir=real_dir, use_tqdm=True)
        candidates = ds.load()

        if not candidates:
            raise FileNotFoundError(
                f"No PNG candidates found in: {real_dir}\n"
                "Check paths.real_png_dir in config/default.yaml."
            )

        for c in candidates:
            if getattr(c, "band", None) is None:
                c.set_band(self._classify_band(c))

        return candidates
    
    def _load_val_embedding(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load the UMAP embedding of the simulated validation set if available.

        These points serve as the background cloud in the plot, providing
        context on where the 64 ON/OFF categories are distributed in the
        learned latent space.
        """
        z_path = self._kde_dir / "umap_val_points.npy"
        y_path = self._kde_dir / "umap_val_labels.npy"
        if z_path.exists() and y_path.exists():
            return np.load(z_path), np.load(y_path)
        return None, None

    def _add_background(self, fig: go.Figure, max_points: int) -> go.Figure:
        """
        Add the simulated validation set as a background scatter layer.

        The three category groups (mixed/noise, only-ON, only-OFF) are
        rendered as separate traces so they can be toggled in the legend.
        """
        Z_bg, y_bg = self._Z_val, self._y_val

        # Subsample for rendering performance when the validation set is large
        if len(Z_bg) > max_points:
            idx = np.random.choice(len(Z_bg), max_points, replace=False)
            Z_bg, y_bg = Z_bg[idx], y_bg[idx]

        mask_on  = y_bg == self._only_on_cat
        mask_off = y_bg == self._only_off_cat
        mask_mix = ~mask_on & ~mask_off

        # Render mixed/noise first so real candidates appear on top
        fig.add_trace(go.Scattergl(
            x=Z_bg[mask_mix, 0], y=Z_bg[mask_mix, 1],
            mode="markers",
            marker=dict(size=2, color="rgba(150,150,170,0.12)"),
            name="Simulated (mixed/noise)",
            hoverinfo="skip",
        ))

        if mask_off.any():
            fig.add_trace(go.Scattergl(
                x=Z_bg[mask_off, 0], y=Z_bg[mask_off, 1],
                mode="markers",
                marker=dict(size=3, color="rgba(255,80,100,0.35)"),
                name="Simulated only-OFF",
                hoverinfo="skip",
            ))

        if mask_on.any():
            fig.add_trace(go.Scattergl(
                x=Z_bg[mask_on, 0], y=Z_bg[mask_on, 1],
                mode="markers",
                marker=dict(size=3, color="rgba(100,255,80,0.35)"),
                name="Simulated only-ON",
                hoverinfo="skip",
            ))

        return fig

    def _add_candidates(
        self, fig: go.Figure, highlight_id: Optional[str]
    ) -> go.Figure:
        """
        Add real candidates as colored markers, grouped by observing band.

        Each band is a separate trace to allow toggling in the legend.
        The highlighted candidate is added last so it renders on top.
        """
        for band, color in self._BAND_COLORS.items():
            idxs = [
                i for i, c in enumerate(self._candidates)
                if getattr(c, "band", "OUT_OF_BAND") == band
            ]
            if not idxs:
                continue

            fig.add_trace(go.Scatter(
                x=self._Z[idxs, 0],
                y=self._Z[idxs, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(width=1.5, color="white"),
                    symbol="circle",
                ),
                name=f"Real — {band} band",
                text=[self._hover_text(self._candidates[i], self._Z[i]) for i in idxs],
                customdata=[self._candidates[i].id for i in idxs],
                hovertemplate="%{text}<extra></extra>",
            ))

        # Highlighted candidate rendered last (on top of all other traces)
        if highlight_id is not None:
            for i, c in enumerate(self._candidates):
                if c.id == highlight_id:
                    fig.add_trace(go.Scatter(
                        x=[self._Z[i, 0]],
                        y=[self._Z[i, 1]],
                        mode="markers",
                        marker=dict(
                            size=22,
                            color="yellow",
                            symbol="star",
                            line=dict(width=2, color="black"),
                        ),
                        name=f"★ {highlight_id}",
                        hovertemplate=self._hover_text(c, self._Z[i]) + "<extra></extra>",
                    ))
                    break

        return fig

    @staticmethod
    def _hover_text(c: Candidate, z: np.ndarray) -> str:
        """Build the HTML hover tooltip string for a candidate."""
        lines = [
            f"<b>{c.id}</b>",
            f"Target: {getattr(c, 'target', 'UNKNOWN')}",
            f"Band: {getattr(c, 'band', '?')}",
            f"Freq: {c.frequency_hz / 1e6:.3f} MHz",
            f"Drift: {c.drift_hz_s:.4f} Hz/s",
            f"UMAP: ({z[0]:.2f}, {z[1]:.2f})",
        ]
        if c.density_score is not None:
            lines.append(f"Density score: {c.density_score:.4e}")
        if getattr(c, "frequency_score", None) is not None:
            lines.append(f"Freq score: {c.frequency_score:.4f}")
        if getattr(c, "similarity_score", None) is not None:
            lines.append(f"Sim score: {c.similarity_score:.4f}")
        return "<br>".join(lines)
    
    @staticmethod
    def _classify_band(candidate: Candidate) -> str:
        """
        Assign a band label based on the candidate central frequency.

        SRT receivers:
            C band:  4.2 - 7.7 GHz
            K band: 18.0 - 26.5 GHz
        """
        f = candidate.frequency_hz
        if 4.2e9 <= f <= 7.7e9:
            return "C"
        if 18.0e9 <= f <= 26.5e9:
            return "K"
        return "OUT_OF_BAND"