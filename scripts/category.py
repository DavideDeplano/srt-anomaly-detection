import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
from pathlib import Path

def create_category_report(candidates, output_filename="results/REPORT_CATEGORIES.pdf"):
    """
    Generate a PDF report grouping candidates by assigned category.

    The function reads `candidate.category` and visualizes one panel per candidate,
    arranged in a fixed grid layout, to allow quick visual inspection of categories.
    """
    path = Path(output_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- PDF Report Generation: {path} ---")

    # 1) Keep only candidates that have been assigned a category
    valid_candidates = [c for c in candidates if c.category is not None]
    
    if not valid_candidates:
        print("WARNING: No candidates have an assigned category.")
        return

     # 2) Sort by category, then by descending density score
    #    This groups categories together and shows strongest candidates first
    valid_candidates.sort(key=lambda x: (x.category, -(x.density_score or 0)))

    print(f"Generating layout for {len(valid_candidates)} classified candidates...")

    # Grid configuration: 25 images per page (5 rows x 5 columns)
    ROWS, COLS = 5, 5
    items_per_page = ROWS * COLS
    num_pages = math.ceil(len(valid_candidates) / items_per_page)

    with PdfPages(path) as pdf:
        for i in range(num_pages):
            # Select candidates for the current page
            batch = valid_candidates[i*items_per_page : (i+1)*items_per_page]
            
            # Create page figure
            fig, axes = plt.subplots(ROWS, COLS, figsize=(20, 24))
            axes = axes.flatten()
            
            # Page header with category range
            cat_start = batch[0].category
            cat_end = batch[-1].category
            fig.suptitle(f"Category Report - Page {i+1}/{num_pages} (Cat {cat_start} -> {cat_end})", fontsize=20)
            
            for idx, cand in enumerate(batch):
                ax = axes[idx]
                
                # Extract first panel of the cadence for visualization
                raw = cand.cadence
                if raw.ndim == 3: 
                    img = raw[0, :, :] # First panel
                else: 
                    img = raw
                
                # Contrast normalization using percentiles
                vmin, vmax = np.nanpercentile(img, 1), np.nanpercentile(img, 99)
                img_norm = np.clip((img - vmin) / (vmax - vmin + 1e-9), 0, 1)
                
                ax.imshow(img_norm, aspect='auto', cmap='viridis')
                
                # Labels and metadata
                cat = cand.category
                score = cand.density_score if cand.density_score is not None else 0.0
                
                name = Path(cand.source_path).name[-15:] # Last 15 chars of filename
                
                ax.set_title(f"CAT: {cat}\nScore: {score:.1e}\n{name}", 
                             fontsize=9, weight='bold', backgroundcolor='#f0f0f0')
                ax.axis('off')
            
            # Disable unused grid cells on the last page
            for j in range(idx + 1, len(axes)):
                axes[j].axis('off')
                
            pdf.savefig(fig)
            plt.close(fig)
            
    print("Report generated successfully!")