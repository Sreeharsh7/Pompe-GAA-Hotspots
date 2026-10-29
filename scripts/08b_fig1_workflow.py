"""Stage 8b: Fig. 1 workflow schematic, drawn with the DI matrix and SSS values from the analysis.

Icons are read from data/raw/figure_icons/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, Rectangle

from gaa_hotspot import figures as F
from gaa_hotspot.common import AA, path, read_processed, table
from gaa_hotspot.plotting import save

ASSETS = path("raw") / "figure_icons"
W, H = 14.0, 7.0
plt.rcParams.update({"font.family": "Arial", "savefig.dpi": 600})


def icon(ax, name, x0, x1, y0, y1):
    img = mpimg.imread(ASSETS / f"{name}.png")
    h, w = img.shape[:2]
    # keep aspect ratio inside the box, centred
    bw, bh = x1 - x0, y1 - y0
    s = min(bw / w, bh / h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.imshow(img, extent=(cx - w * s / 2, cx + w * s / 2, cy - h * s / 2, cy + h * s / 2), zorder=2,
              interpolation="lanczos")


def arrow(ax, pts, dashed=True, lw=1.8):
    ls = (0, (4, 3)) if dashed else "-"
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), alpha=0)  # keep limits
    for (a, b) in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=lw, ls=ls, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=18, color="black", lw=lw,
                                 linestyle=ls if dashed else "-", shrinkA=0, shrinkB=0))


def cylinder(ax, x, y, w, seg, n, fill="#FFD25E", dark="#F2B533", edge="#1c1c1c", lw=2.6):
    eh = w * 0.32
    top = y + seg * n
    ax.add_patch(Rectangle((x, y), w, seg * n, facecolor=fill, edgecolor="none", zorder=3))
    ax.add_patch(Ellipse((x + w / 2, y), w, eh, facecolor=fill, edgecolor=edge, lw=lw, zorder=3))
    ax.add_patch(Rectangle((x, y), w, seg * n, facecolor=fill, edgecolor="none", zorder=3.1))
    ax.add_patch(Rectangle((x + w * 0.62, y), w * 0.38, seg * n, facecolor=dark, edgecolor="none", zorder=3.1, alpha=0.6))
    ax.plot([x, x], [y, top], color=edge, lw=lw, zorder=3.2)
    ax.plot([x + w, x + w], [y, top], color=edge, lw=lw, zorder=3.2)
    for k in range(n):
        yy = y + seg * k
        ax.add_patch(Arc((x + w / 2, yy), w, eh, theta1=180, theta2=360, color=edge, lw=lw, zorder=3.3))
        ax.plot([x + w * 0.42, x + w * 0.58], [yy + seg * 0.42] * 2, color=edge, lw=lw * 1.3,
                solid_capstyle="round", zorder=3.3)
    ax.add_patch(Ellipse((x + w / 2, top), w, eh, facecolor="#FFE08A", edgecolor=edge, lw=lw, zorder=3.4))


def main():
    di = pd.read_csv(table("di_matrix.csv")).set_index("aa")
    sss = read_processed("sss_all_substitutions.csv")
    feat = read_processed("residue_features.csv").set_index("residue")
    novel = pd.read_csv(table("novel_hotspots.csv")).set_index("residue")

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    bold = dict(fontweight="bold", ha="center", va="center")

    # inputs
    ax.text(1.05, 6.72, "Residue Interaction\nNetwork (RIN)", fontsize=13, **bold)
    icon(ax, "rin", 0.25, 1.85, 4.95, 6.3)
    ax.add_patch(FancyArrowPatch((1.05, 4.45), (1.05, 4.92), arrowstyle="-|>", mutation_scale=22, color="black", lw=3))
    ax.text(1.05, 4.22, "GAA Structure", fontsize=13, **bold)
    icon(ax, "struct", 0.15, 1.95, 2.35, 4.05)
    ax.text(1.35, 1.72, "Multiple Sequence Alignment", fontsize=13, **bold)
    icon(ax, "msa", 0.15, 2.55, 0.25, 1.5)

    # feature dataset
    cylinder(ax, 3.62, 3.0, 0.6, 0.38, 3)
    cylinder(ax, 3.1, 2.62, 0.6, 0.38, 3)
    ax.add_patch(Ellipse((3.4, 2.62), 0.6, 0.19, facecolor="#FFD25E", edgecolor="#1c1c1c", lw=2.6, zorder=2.9))
    ax.text(4.32, 2.45, "Feature\nDataset", fontsize=13, fontweight="bold", ha="left", va="center")
    arrow(ax, [(1.9, 5.65), (3.92, 5.65), (3.92, 4.3)])
    ax.text(4.12, 5.0, "Network\nFeatures", fontsize=12, style="italic", rotation=-90, ha="center", va="center")
    arrow(ax, [(1.95, 3.25), (3.05, 3.25)])
    ax.text(2.5, 3.6, "Structural\nFeatures", fontsize=12, style="italic", ha="center", va="center")
    arrow(ax, [(2.6, 0.85), (3.4, 0.85), (3.4, 2.45)])
    ax.text(3.65, 1.55, "Evolutionary\nscores", fontsize=12, style="italic", rotation=-90, ha="center", va="center")

    # model
    arrow(ax, [(4.45, 3.25), (5.3, 3.25)], dashed=False, lw=3)
    icon(ax, "nn", 5.35, 6.85, 2.55, 3.95)
    ax.text(6.1, 2.25, "ML Model\n(5-classifier ensemble)", fontsize=12, **bold)
    icon(ax, "hosp", 5.6, 6.6, 5.95, 6.95)
    ax.text(6.1, 5.55, "GAA Clinical Data", fontsize=13, **bold)
    arrow(ax, [(6.1, 5.25), (6.1, 4.05)])
    arrow(ax, [(6.95, 3.25), (7.4, 3.25)], dashed=False, lw=3)
    ax.text(8.3, 3.25, "Critical Residue\nPositions", fontsize=16, **bold)

    # substitution scoring
    arrow(ax, [(8.3, 3.75), (8.3, 6.0), (8.95, 6.0)])
    icon(ax, "logo", 9.0, 11.9, 5.55, 6.5)
    ax.text(10.45, 5.25, "Evolutionary Constraint Score", fontsize=13, **bold)
    arrow(ax, [(8.3, 2.75), (8.3, 1.6), (9.05, 1.6)])
    dax = fig.add_axes([9.25 / W, 0.75 / H, 2.0 / W, 1.75 / H])
    mat = (di / di.values.max()).loc[AA, AA].values.copy()
    mat[np.triu_indices(20, 1)] = np.nan
    dax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    dax.set_xticks(range(20))
    dax.set_xticklabels(AA, fontsize=5.5)
    dax.set_yticks(range(20))
    dax.set_yticklabels(AA, fontsize=5.5)
    dax.tick_params(length=0, pad=1)
    for s in dax.spines.values():
        s.set_visible(False)
    ax.text(10.3, 0.3, "Physicochemical Constraint Score (DI)", fontsize=13, **bold)

    hax = fig.add_axes([11.75 / W, 3.3 / H, 2.05 / W, 1.45 / H])
    hm = F.sss_matrix(sss, sorted(novel.index)[:30], feat).values
    hax.set_facecolor("#d9d9d9")
    hax.imshow(hm, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    hax.set_xticks([])
    hax.set_yticks(range(20))
    hax.set_yticklabels(AA, fontsize=4.5)
    hax.tick_params(length=0, pad=1)
    for s in hax.spines.values():
        s.set_visible(False)
    ax.text(12.75, 3.0, "Substitution Severity\nScore (SSS) Heatmap", fontsize=13, **bold)
    arrow(ax, [(11.95, 6.0), (12.75, 6.0), (12.75, 4.85)])
    arrow(ax, [(11.55, 1.6), (12.75, 1.6), (12.75, 2.6)])

    save(fig, path("figures_main"), "Fig1_workflow")


if __name__ == "__main__":
    main()
