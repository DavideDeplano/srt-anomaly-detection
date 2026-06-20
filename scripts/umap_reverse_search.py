"""
UMAPReverseSearch

Given a real candidate, project it into the trained UMAP space and
visualize where it falls relative to all other candidates.

Useful for:
- Understanding which cluster a candidate belongs to
- Finding morphologically similar candidates in the same UMAP region
- Investigating whether multiple candidates correspond to the same RFI type

Interactive HTML features
-------------------------
- Click on a point to show metadata + source file path in the right panel.
- Zoom with the mouse wheel; pan by dragging.
- The legend is display-only.
- No images embedded — HTML stays lightweight regardless of dataset size.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, Div
from bokeh.layouts import row
from bokeh.embed import file_html
from bokeh.resources import INLINE

from src.srtad.config import filters, paths
from src.srtad.core.candidate import Candidate
from src.srtad.core.dataset import Dataset
from src.srtad.management.cross_correlation_extractor import CrossCorrelationExtractor


class UMAPReverseSearch:
    """
    Projects real SRT candidates into the trained UMAP density space
    and provides interactive visualization and nearest-neighbor search.

    Attributes
    ----------
    _umap_model  : fitted UMAP model loaded from disk
    _candidates  : list of real Candidate objects successfully projected
    _passed_ids  : set of candidate IDs that passed the density filter
    _Z           : (N, 2) UMAP coordinates for each candidate in _candidates
    _Z_val       : (M, 2) UMAP coordinates of the simulated validation set
    _y_val       : (M,)   category labels of the simulated validation set
    """

    _BAND_COLORS: Dict[str, str] = {
        "C": "#00e5ff",
        "K": "#bd93f9",
        "OUT_OF_BAND": "#888888",
    }

    def __init__(self) -> None:
        self._umap_path   = Path(filters["density"]["umap_model_path"])
        self._kde_dir     = Path(filters["density"]["kde_models_dir"])
        self._results_dir = Path(paths["results"])

        self._only_on_cat  = int(filters["density"]["only_on_category"])
        self._only_off_cat = int(filters["density"]["only_off_category"])

        self._extractor  = CrossCorrelationExtractor()
        self._umap_model = None
        self._candidates: List[Candidate] = []
        self._passed_ids: Set[str] = set()
        self._Z:     Optional[np.ndarray] = None
        self._Z_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None

    def load(self) -> "UMAPReverseSearch":
        print("[1/3] Loading UMAP model...")
        self._umap_model = self._load_umap()

        print("[2/3] Loading real candidates...")
        self._candidates, self._passed_ids = self._load_candidates()
        print(f"      → {len(self._candidates)} total candidates loaded")
        print(f"      → {len(self._passed_ids)} passed density filter")

        print("[3/3] Loading simulated validation background...")
        self._Z_val, self._y_val = self._load_val_embedding()
        if self._Z_val is not None:
            print(f"      → {len(self._Z_val)} validation points loaded")
        else:
            print("      → validation background not available")

        return self

    def project(self) -> np.ndarray:
        self._check_loaded()

        print(f"Projecting {len(self._candidates)} candidates into UMAP space...")

        features: List[np.ndarray] = []
        valid_candidates: List[Candidate] = []
        n_skipped = 0

        for c in self._candidates:
            cadence = getattr(c, "cadence", None)
            if cadence is None:
                n_skipped += 1
                continue
            try:
                feat = self._extractor.extract_features(cadence)
                features.append(feat)
                valid_candidates.append(c)
            except Exception as exc:
                print(f"[WARN] Skipping {c.id}: {type(exc).__name__}: {exc}")
                n_skipped += 1

        if n_skipped > 0:
            print(
                f"[WARN] {n_skipped}/{len(self._candidates)} candidates skipped "
                f"(missing cadence or extraction error)."
            )

        if not features:
            raise RuntimeError(
                "No valid candidates to project — all cadences are missing or "
                "failed extraction. Cannot build UMAP plot."
            )

        self._candidates = valid_candidates
        X = np.array(features, dtype=np.float64)
        self._Z = self._umap_model.transform(X)
        print(f"Projection complete: {len(self._candidates)} candidates, shape {self._Z.shape}")
        return self._Z

    def plot(
        self,
        highlight_id: Optional[str] = None,
        max_bg_points: int = 30_000,
        save_html: bool = True,
    ) -> None:
        if self._Z is None:
            self.project()

        # ---- Bokeh figure ----
        p = figure(
            width=950,
            height=780,
            title="UMAP Reverse Search — Real SRT Candidates",
            tools="pan,wheel_zoom,reset,tap",
            active_scroll="wheel_zoom",
            active_drag="pan",
            output_backend="webgl",
        )
        p.background_fill_color        = "#0b0c10"
        p.border_fill_color            = "#080a0f"
        p.title.text_color             = "#ccd6f6"
        p.title.text_font_size         = "16px"
        p.xaxis.axis_label             = "UMAP X"
        p.yaxis.axis_label             = "UMAP Y"
        p.xaxis.axis_label_text_color  = "#8892b0"
        p.yaxis.axis_label_text_color  = "#8892b0"
        p.xaxis.major_label_text_color = "#8892b0"
        p.yaxis.major_label_text_color = "#8892b0"
        p.xgrid.grid_line_color        = "#1c2035"
        p.ygrid.grid_line_color        = "#1c2035"
        p.toolbar.logo                 = None

        # ---- Background: simulated validation set ----
        if self._Z_val is not None and self._y_val is not None:
            Z_bg, y_bg = self._Z_val, self._y_val
            if len(Z_bg) > max_bg_points:
                idx  = np.random.choice(len(Z_bg), max_bg_points, replace=False)
                Z_bg = Z_bg[idx]
                y_bg = y_bg[idx]

            mask_on  = y_bg == self._only_on_cat
            mask_off = y_bg == self._only_off_cat
            mask_mix = ~mask_on & ~mask_off

            if mask_mix.any():
                p.scatter("x", "y",
                          source=ColumnDataSource(dict(x=Z_bg[mask_mix,0].tolist(),
                                                       y=Z_bg[mask_mix,1].tolist())),
                          size=2, color="#969696", alpha=0.08,
                          legend_label="Simulated (mixed/noise)")
            if mask_off.any():
                p.scatter("x", "y",
                          source=ColumnDataSource(dict(x=Z_bg[mask_off,0].tolist(),
                                                       y=Z_bg[mask_off,1].tolist())),
                          size=3, color="#ff5064", alpha=0.25,
                          legend_label="Simulated only-OFF")
            if mask_on.any():
                p.scatter("x", "y",
                          source=ColumnDataSource(dict(x=Z_bg[mask_on,0].tolist(),
                                                       y=Z_bg[mask_on,1].tolist())),
                          size=3, color="#64ff50", alpha=0.25,
                          legend_label="Simulated only-ON")

        # ---- Candidate payload (metadata + source path only, no images) ----
        payload: Dict[str, dict] = {}
        for i, c in enumerate(self._candidates):
            passed = c.id in self._passed_ids
            payload[c.id] = {
                "passed":           passed,
                "target":           getattr(c, "target", "UNKNOWN"),
                "band":             getattr(c, "band", "?"),
                "freq_mhz":         c.frequency_hz / 1e6,
                "drift_hz_s":       float(c.drift_hz_s),
                "umap_x":           float(self._Z[i, 0]),
                "umap_y":           float(self._Z[i, 1]),
                "density_score":    c.density_score,
                "frequency_score":  getattr(c, "frequency_score", None),
                "similarity_score": getattr(c, "similarity_score", None),
                "source_path":      str(getattr(c, "source_path", "")),
            }

        # ---- Side panel Div ----
        panel_div = Div(
            text="""
            <div style="width:380px;background:#0d0f16;border-left:1px solid #1c2035;
                        padding:18px 16px;font-family:monospace;color:#8892b0;
                        font-size:12px;text-align:center;padding-top:40px;">
                Click a point to inspect
            </div>
            """,
            width=400,
            height=780,
            styles={"background": "#0d0f16", "border-left": "1px solid #1c2035"},
        )

        # ---- Real candidates per band ----
        for band, color in self._BAND_COLORS.items():
            idxs = [
                i for i, c in enumerate(self._candidates)
                if getattr(c, "band", "OUT_OF_BAND") == band
            ]
            if not idxs:
                continue

            src = ColumnDataSource(dict(
                x   = [float(self._Z[i, 0]) for i in idxs],
                y   = [float(self._Z[i, 1]) for i in idxs],
                cid = [self._candidates[i].id for i in idxs],
            ))

            p.scatter(
                "x", "y",
                source=src,
                size=10,
                color=color,
                line_color="white",
                line_width=1.5,
                legend_label=f"Real — {band} band",
                name=f"band_{band}",
            )

            cb = CustomJS(
                args=dict(source=src, panel=panel_div, payload=payload),
                code="""
                const indices = source.selected.indices;
                if (indices.length === 0) return;

                const cid = source.data['cid'][indices[0]];
                const d   = payload[cid];
                if (!d) return;

                const fmt = v =>
                    (v !== null && v !== undefined) ? Number(v).toExponential(3) : '—';

                panel.text = `
                <div style="width:380px;background:#0d0f16;border-left:1px solid #1c2035;
                             padding:18px 16px;font-family:monospace;color:#ccd6f6;
                             overflow-y:auto;">
                    <div style="font-size:11px;font-weight:bold;
                                color:${d.passed ? '#64ffda' : '#ff5064'};
                                margin-bottom:4px;">
                        ${d.passed ? '✓ PASSED density filter' : '✗ rejected'}
                    </div>
                    <div style="font-size:12px;font-weight:bold;color:#ccd6f6;
                                word-break:break-all;margin-bottom:12px;">${cid}</div>
                    <table style="font-size:12px;border-collapse:collapse;width:100%;
                                  margin-bottom:8px;">
                        <tr>
                            <td style="color:#8892b0;padding:3px 8px 3px 0;
                                       white-space:nowrap;">Target</td>
                            <td>${d.target || 'UNKNOWN'}</td>
                        </tr>
                        <tr>
                            <td style="color:#8892b0;padding:3px 8px 3px 0;
                                       white-space:nowrap;">Band</td>
                            <td>${d.band || '?'}</td>
                        </tr>
                        <tr>
                            <td style="color:#8892b0;padding:3px 8px 3px 0;
                                       white-space:nowrap;">Frequency</td>
                            <td>${d.freq_mhz.toFixed(3)} MHz</td>
                        </tr>
                        <tr>
                            <td style="color:#8892b0;padding:3px 8px 3px 0;
                                       white-space:nowrap;">Drift</td>
                            <td>${d.drift_hz_s.toFixed(4)} Hz/s</td>
                        </tr>
                        <tr>
                            <td style="color:#8892b0;padding:3px 8px 3px 0;
                                       white-space:nowrap;">Density</td>
                            <td>${fmt(d.density_score)}</td>
                        </tr>
                        <tr>
                            <td style="color:#8892b0;padding:3px 8px 3px 0;
                                       white-space:nowrap;">UMAP</td>
                            <td>(${d.umap_x.toFixed(2)}, ${d.umap_y.toFixed(2)})</td>
                        </tr>
                    </table>
                    <div style="padding:8px;background:#111520;border:1px solid #1c2035;
                                border-radius:4px;word-break:break-all;font-size:11px;
                                color:#8892b0;">
                        <span style="color:#64ffda;">Path:</span><br/>
                        ${d.source_path || '—'}
                    </div>
                </div>`;
                """,
            )
            src.selected.js_on_change("indices", cb)

        # ---- Highlighted candidate ----
        if highlight_id is not None:
            for i, c in enumerate(self._candidates):
                if c.id == highlight_id:
                    src_hl = ColumnDataSource(dict(
                        x   = [float(self._Z[i, 0])],
                        y   = [float(self._Z[i, 1])],
                        cid = [c.id],
                    ))
                    p.scatter(
                        "x", "y",
                        source=src_hl,
                        size=22,
                        color="yellow",
                        marker="star",
                        line_color="black",
                        line_width=2,
                        legend_label=f"★ {highlight_id}",
                    )
                    break

        # ---- Legend: display-only ----
        p.legend.background_fill_color = "rgba(15,17,23,0.9)"
        p.legend.border_line_color     = "#1c2035"
        p.legend.label_text_color      = "#8892b0"
        p.legend.label_text_font_size  = "11px"
        p.legend.click_policy          = "none"

        # ---- Save ----
        if save_html:
            layout   = row(p, panel_div)
            out_path = self._results_dir / "umap_reverse_search.html"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            html = file_html(layout, resources=INLINE, title="UMAP Reverse Search — SRT")
            out_path.write_text(html, encoding="utf-8")
            print(f"[OK] Interactive plot saved to: {out_path}")

    def find_neighbors(
        self,
        candidate_id: str,
        radius: float = 2.0,
        top_k: int = 10,
    ) -> List[Dict]:
        if self._Z is None:
            self.project()

        target_idx = next(
            (i for i, c in enumerate(self._candidates) if c.id == candidate_id),
            None,
        )
        if target_idx is None:
            print(f"[WARN] Candidate '{candidate_id}' not found.")
            return []

        z_target = self._Z[target_idx]
        dists    = np.linalg.norm(self._Z - z_target, axis=1)

        neighbors = []
        for i, (c, d) in enumerate(zip(self._candidates, dists)):
            if i == target_idx or d > radius:
                continue
            neighbors.append({
                "id":               c.id,
                "distance":         float(d),
                "frequency_mhz":    c.frequency_hz / 1e6,
                "band":             getattr(c, "band", "?"),
                "density_score":    c.density_score,
                "frequency_score":  getattr(c, "frequency_score", None),
                "similarity_score": getattr(c, "similarity_score", None),
                "umap_x":           float(self._Z[i, 0]),
                "umap_y":           float(self._Z[i, 1]),
            })

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
        if self._umap_model is None or not self._candidates:
            raise RuntimeError("Call load() before using this method.")

    def _load_umap(self):
        if not self._umap_path.exists():
            raise FileNotFoundError(
                f"UMAP model not found: {self._umap_path}\n"
                "Train the density filter first (option 2 in main.py)."
            )
        return joblib.load(self._umap_path)

    def _load_candidates(self) -> Tuple[List[Candidate], Set[str]]:
        """
        Load all real candidates from real_png_dir.
        Reads passed_candidates.pkl to know which ones passed the density filter.
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

        passed_ids: Set[str] = set()
        pkl_path = self._results_dir / "passed_candidates.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                passed_candidates, _ = pickle.load(f)
            passed_ids = {c.id for c in passed_candidates}
        else:
            print("[WARN] passed_candidates.pkl not found — all candidates marked as rejected.")
            print("       Run the density filter first (option 3).")

        return candidates, passed_ids

    def _load_val_embedding(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        z_path = self._kde_dir / "umap_val_points.npy"
        y_path = self._kde_dir / "umap_val_labels.npy"
        if z_path.exists() and y_path.exists():
            return np.load(z_path), np.load(y_path)
        return None, None

    @staticmethod
    def _classify_band(candidate: Candidate) -> str:
        f = candidate.frequency_hz
        if 4.2e9 <= f <= 7.7e9:
            return "C"
        if 18.0e9 <= f <= 26.5e9:
            return "K"
        return "OUT_OF_BAND"