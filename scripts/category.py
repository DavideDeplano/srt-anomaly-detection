from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

from src.srtad.core.dataset import Dataset

def create_category_report(
    candidates,
    output_filename="results/REPORT_CATEGORIES.pdf",
    threshold=None,
    rows=2,
    cols=2,
    dpi=300,
    show_all_6_panels=True,
):
    """
    Generate a PDF report of candidates grouped by category.

    For each candidate, the function:
    - loads the original waterfall PNG from `source_path`;
    - applies a fixed crop to isolate the spectrogram region;
    - preprocesses the image using `Dataset._preprocess_spectrogram`;
    - reconstructs the cadence layout (six panels or a single panel);
    - normalizes contrast and renders the result into a grid layout.

    The report is saved as a multi-page PDF at `output_filename`.
    """
    path = Path(output_filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n--- PDF Report Generation: {path} ---")

    ds = Dataset()

    def _passed(c):
        """
        Decide whether a candidate passes the density criterion.

        Priority:
        - If `c.passed_density` exists and is not None, return it as boolean.
        - Otherwise fall back to `c.density_score`:
            * if `threshold` is provided: pass if score >= threshold
            * else: pass if score > 0.0
        """
        if hasattr(c, "passed_density") and c.passed_density is not None:
            return bool(c.passed_density)
        s = getattr(c, "density_score", None)
        if s is None:
            return False
        return s >= threshold if threshold is not None else s > 0.0

    def _crop_box_local(w: int, h: int):
        """
        Compute a crop rectangle using fixed fractional margins.

        The returned tuple is (left, top, right, bottom) in pixel coordinates.
        """
        return (
            int(w * 0.0894),        # left
            int(h * 0.0440),        # top
            int(w * (1 - 0.1490)),  # right
            int(h * (1 - 0.0670)),  # bottom
        )

    # Filter candidates: passed + categorized + source_path available
    valid = [
        c for c in candidates
        if _passed(c)
        and getattr(c, "category", None) is not None
        and getattr(c, "source_path", None) is not None
    ]

    if not valid:
        print("WARNING: no valid candidates.")
        return

    # Sort by category, then by descending density_score
    valid.sort(key=lambda x: (x.category, -(x.density_score or 0.0)))

    per_page = rows * cols
    num_pages = math.ceil(len(valid) / per_page)

    with PdfPages(path) as pdf:
        for i in range(num_pages):
            batch = valid[i * per_page:(i + 1) * per_page]

            # Create page grid
            fig, axes = plt.subplots(rows, cols, figsize=(26, 16))
            axes = np.array(axes).flatten()

            fig.suptitle(
                f"Category Report - Page {i+1}/{num_pages} "
                f"(Cat {batch[0].category} -> {batch[-1].category})",
                fontsize=22
            )

            last_idx = -1

            for idx, cand in enumerate(batch):
                last_idx = idx
                ax = axes[idx]

                # Load and crop the original PNG for this candidate
                try:
                    sp = cand.source_path
                    if isinstance(sp, (list, tuple)):
                        sp = sp[0]
                    sp = Path(str(sp))

                    if not sp.exists():
                        raise FileNotFoundError(sp)

                    with Image.open(sp) as im:
                        im = im.convert("L")
                        w, h = im.size
                        cropped = im.crop(_crop_box_local(w, h))
                        arr = np.asarray(cropped, dtype=np.float64)

                except Exception as e:
                    ax.set_title(
                        f"FAILED TO READ\n{str(getattr(cand,'source_path','NO_PATH'))}\n"
                        f"{type(e).__name__}: {e}",
                        fontsize=9,
                        weight="bold"
                    )
                    ax.axis("off")
                    continue

                # Preprocess (same routine used by the Dataset loader)
                arr = ds._preprocess_spectrogram(arr)

                # Split into 6 vertical panels and optionally stack them
                H = arr.shape[0]
                step = H // 6
                if step <= 0:
                    img = arr
                else:
                    panels = [arr[k * step:(k + 1) * step, :] for k in range(6)]
                    img = np.vstack(panels) if show_all_6_panels else panels[0]

                # Contrast normalization for readability
                vmin, vmax = np.nanpercentile(img, 1), np.nanpercentile(img, 99)
                img = np.clip((img - vmin) / (vmax - vmin + 1e-9), 0, 1)

                # Render
                ax.imshow(img, aspect="auto", cmap="viridis", interpolation="nearest")

                name = sp.name
                score = cand.density_score or 0.0

                ax.set_title(
                    f"CAT: {cand.category} | Score: {score:.3e}\n{name}",
                    fontsize=9,
                    weight="bold",
                    backgroundcolor="#f0f0f0"
                )
                ax.axis("off")

            # Disable unused axes on the last page
            for j in range(last_idx + 1, len(axes)):
                axes[j].axis("off")

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    print("Report generated successfully.")
