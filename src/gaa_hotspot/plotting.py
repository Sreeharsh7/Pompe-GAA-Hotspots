"""Figure style matching the figures of manuscript draft v7.

Arial, bold axis labels, white background with a light dotted grid, no top/right spines,
the matplotlib "tab" colours used in v7 (models, criticality groups, predictors) and
seaborn box/strip plots and heatmaps. Every figure is written as PDF, SVG and 600 dpi PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Model colours and names exactly as in v7 Figure 4a (legend order LR, RF, kNN, SVM, XGB).
MODEL_COLOURS = {"lr": "#1f77b4", "rf": "#2ca02c", "knn": "#ff7f0e", "svm": "#9467bd", "xgb": "#d62728"}
MODEL_LABELS = {"lr": "Logistic Regression", "rf": "Random Forest", "knn": "k-Nearest Neighbors",
                "svm": "Support Vector Machine", "xgb": "XGBoost"}
MODEL_SHORT = {"lr": "LR", "rf": "RF", "svm": "SVM", "xgb": "XGB", "knn": "kNN"}
V7_MODEL_ORDER = ["lr", "rf", "knn", "svm", "xgb"]
# Bar-chart order and Set2 colours of the v7 "model results" figure.
BAR_MODEL_ORDER = ["rf", "svm", "lr", "knn", "xgb"]
BAR_COLOURS = {"rf": "#66c2a5", "svm": "#fc8d62", "lr": "#8da0cb", "knn": "#e78ac3", "xgb": "#a6d854"}

# Criticality groups as in v7 Figures 3a and 4c,d.
GROUP_COLOURS = {"HDHB": "#2ca02c", "LDHB": "#d62728", "LDLB": "#1f77b4", "SC": "#ff7f0e", "None": "#a9a9a9"}
GROUP_ORDER = ["HDHB", "LDHB", "LDLB", "SC"]

# Mutation / no-mutation boxplots of v7 Supplementary Figure 1.
LABEL_COLOURS = {0: "#1f77b4", 1: "#ff7f0e"}

# Predictors as in v7 Figure 6a (colour, line style).
TOOL_STYLE = {
    "AlphaMissense": ("#1f77b4", "-"),
    "SSS": ("#d62728", "-"),
    "ESM-1v": ("#2ca02c", "--"),
    "EVE": ("#ff7f0e", "-."),
    "EVOscore": ("#9467bd", ":"),
    "DI": ("#8c564b", ":"),
}
TOOL_LABELS = {"SSS": "Combined SSS [This Study]", "AlphaMissense": "AlphaMissense", "ESM-1v": "ESM-1v", "EVE": "EVE",
               "EVOscore": "EVOscore (SSS term)", "DI": "DI (SSS term)"}
CLASS_COLOURS = {"Benign": "#2ca02c", "Pathogenic": "#d62728"}

FEATURE_LABELS = {
    "degree": "degree", "log_betweenness": "log_betweenness", "betweenness": "betweenness", "closeness": "closeness",
    "burt": "burt", "authority": "authority", "page_rank": "page_rank", "k_core": "k_core", "kdhydro": "kdhydro",
    "areaSAS": "areaSAS", "areaSES": "areaSES", "relSESA": "relSESA", "phi": "phi", "psi": "psi",
    "conservation": "Conserv_score",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#595959",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "figure.dpi": 110,
        }
    )


def tidy(ax, grid: str = "both") -> None:
    """v7 axis finish: dotted light grid, grey left/bottom spines."""
    if grid:
        ax.grid(True, axis=grid, linestyle=":", alpha=0.3, color="grey")
        ax.set_axisbelow(True)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left")


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
