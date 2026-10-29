"""Figure panels in the style of manuscript draft v7.

Each function returns a matplotlib Figure. scripts/08_figures.py saves them and the
notebooks call the same functions, so the figures shown in the notebooks are identical
to the files in results/figures/. Protein-structure renderings (Fig. 2a, 3b, 3d, 4b) are
made in PyMOL/ChimeraX from results/structures/ and are not drawn here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
from scipy.spatial.distance import cdist
from sklearn.metrics import auc, confusion_matrix, roc_curve

from .common import AA
from .plotting import (BAR_COLOURS, BAR_MODEL_ORDER, CLASS_COLOURS, FEATURE_LABELS, GROUP_COLOURS, GROUP_ORDER,
                       LABEL_COLOURS, MODEL_COLOURS, MODEL_LABELS, MODEL_SHORT, TOOL_LABELS, TOOL_STYLE,
                       V7_MODEL_ORDER, panel_label, tidy)


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def sci(p: float) -> str:
    """P-value as mathtext, e.g. 1.62 x 10^-8."""
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.2e}".split("e")
    return rf"${m} \times 10^{{{int(e)}}}$"


def plot_groups(groups: pd.DataFrame) -> pd.Series:
    """Criticality label per residue with the 10 supercritical hubs shown as 'SC' (as in v7 Fig. 4c,d)."""
    g = groups["criticality"].replace({"Other": "None"}).copy()
    g[groups["supercritical"].astype(bool)] = "SC"
    return g


# =========================================================================================
# Figure 3 — topological criticality
# =========================================================================================

def fig2b_centrality_correlation(corr: pd.DataFrame) -> plt.Figure:
    """v7 Fig. 2b: absolute Spearman correlations between the seven RIN centralities (lower triangle)."""
    labels = {"betweenness": "Betweenness", "page_rank": "PageRank", "degree": "Degree", "burt": "Burt's constraint",
              "k_core": "K-core", "closeness": "Closeness", "authority": "Authority"}
    mat = corr.abs()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sns.heatmap(mat, mask=np.triu(np.ones_like(mat, dtype=bool), k=1), cmap="Reds", vmin=0, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 10}, square=True, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "|Spearman ρ|", "shrink": 0.8}, ax=ax)
    ax.set_xticklabels([labels[c] for c in mat.columns], rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels([labels[c] for c in mat.index], rotation=0, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def fig3a_criticality(feat: pd.DataFrame, groups: pd.DataFrame, edges: dict) -> plt.Figure:
    df = feat[["degree", "log_betweenness"]].copy()
    df["Criticality"] = groups["criticality"].replace({"Other": "None"})
    order = ["LDLB", "None", "HDHB", "LDHB"]
    counts = df["Criticality"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="degree", y="log_betweenness", hue="Criticality", hue_order=order,
                    palette=GROUP_COLOURS, s=45, edgecolor="w", linewidth=0.5, alpha=0.9, ax=ax)
    for e in edges["log_betweenness"][1:-1]:
        ax.axhline(e, ls="--", color="#555555", lw=0.8, alpha=0.6)
    for e in edges["degree"][1:-1]:
        ax.axvline(e, ls="--", color="#555555", lw=0.8, alpha=0.6)
    tidy(ax)
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.set_xlabel("Degree (Connectivity)")
    ax.set_ylabel("Log Betweenness Centrality")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [f"{l} (n={counts.get(l, 0)})" for l in labels], title="Criticality",
              loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    return fig


def fig3c_supercritical_3d(feat: pd.DataFrame, groups: pd.DataFrame) -> plt.Figure:
    x = feat[["degree", "log_betweenness", "closeness"]]
    ideal = x.max().values
    dist = cdist(x.values, ideal.reshape(1, -1)).ravel()
    sc = groups["supercritical"].astype(bool).values
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    scat = ax.scatter(x.degree, x.log_betweenness, x.closeness, c=dist, cmap="viridis_r", s=30,
                      edgecolors="k", linewidths=0.4, alpha=0.8)
    ax.scatter(x.degree[sc], x.log_betweenness[sc], x.closeness[sc], facecolors="none", edgecolors="#ff7f0e",
               s=110, linewidths=1.8, label="Supercritical Trade-offs (Top 10)", depthshade=False)
    ax.scatter(*ideal, color="#2ca02c", marker="X", s=150, edgecolors="k", linewidths=0.5,
               label="Theoretical Ideal Point", depthshade=False)
    ax.set_xlabel("Degree (Contacts)", labelpad=8, fontweight="bold")
    ax.set_ylabel("Log Betweenness Centrality", labelpad=8, fontweight="bold")
    ax.set_zlabel("Closeness Centrality", labelpad=8, fontweight="bold")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0.95, 0.95, 0.95, 0.5))
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=8)
    cbar = fig.colorbar(scat, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Euclidean Distance to Ideal")
    cbar.ax.tick_params(labelsize=8)
    return fig


def fig3e_venn(feat: pd.DataFrame, groups: pd.DataFrame) -> plt.Figure:
    mut = set(feat.index[feat.mutation_observed == 1])
    hdhb = set(groups.index[groups.criticality == "HDHB"])
    hubs = set(groups.index[groups.supercritical.astype(bool)])
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = (f"HGMD Missense Mutation Positions\n(Total: {len(mut)})", f"HDHB Residues\n(Total: {len(hdhb)})",
              f"SC Hub Residues\n(Total: {len(hubs)})")
    v = venn3([mut, hdhb, hubs], set_labels=labels, ax=ax)
    style = {"100": ("#1f77b4", 0.4), "010": ("#2ca02c", 0.4), "001": ("#d62728", 0.4), "110": ("#1f77b4", 0.45),
             "011": ("#2ca02c", 0.45), "101": ("#d62728", 0.45), "111": ("#d62728", 0.55)}
    for rid, (c, a) in style.items():
        patch = v.get_patch_by_id(rid)
        if patch is not None:
            patch.set_color(c)
            patch.set_alpha(a)
    for rid in style:
        lbl = v.get_label_by_id(rid)
        if lbl is not None:
            lbl.set_fontsize(10)
            lbl.set_fontweight("bold")
    for rid in "ABC":
        lbl = v.get_label_by_id(rid)
        if lbl is not None:
            lbl.set_fontsize(10)
            lbl.set_fontweight("bold")
    for c in venn3_circles([mut, hdhb, hubs], linestyle="solid", linewidth=0.8, ax=ax):
        c.set_edgecolor("#555555")
    return fig


# =========================================================================================
# Figure 4 — ensemble classifiers and criticality-group profiles
# =========================================================================================

def fig4a_roc(prob: pd.DataFrame) -> plt.Figure:
    y = prob["mutation_observed"].values
    fig, ax = plt.subplots(figsize=(6, 5))
    tidy(ax)
    for m in V7_MODEL_ORDER:
        fpr, tpr, _ = roc_curve(y, prob[m].values)
        ax.plot(fpr, tpr, color=MODEL_COLOURS[m], lw=1.5, label=f"{MODEL_LABELS[m]} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#7f7f7f", lw=1.0, label="Random Prediction (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate", labelpad=8)
    ax.set_ylabel("True Positive Rate", labelpad=8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig4_group_profile(feat: pd.DataFrame, groups: pd.DataFrame, column: str, ylabel: str) -> plt.Figure:
    df = pd.DataFrame({"criticality": plot_groups(groups), "value": feat[column]})
    df = df[df.criticality != "None"]
    n = df.criticality.value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    tidy(ax)
    sns.boxplot(data=df, x="criticality", y="value", hue="criticality", order=GROUP_ORDER, palette=GROUP_COLOURS,
                fliersize=0, boxprops=dict(alpha=0.3), legend=False, ax=ax)
    sns.stripplot(data=df, x="criticality", y="value", hue="criticality", order=GROUP_ORDER, palette=GROUP_COLOURS,
                  edgecolor="black", alpha=0.7, size=4, linewidth=0.5, jitter=0.12, legend=False, ax=ax)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([f"{g}\n(n={n.get(g, 0)})" for g in GROUP_ORDER])
    ax.set_xlabel("Topological Criticality Groups", labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    fig.tight_layout()
    return fig


# =========================================================================================
# Figure 5 — Substitution Severity Score
# =========================================================================================

def _box(ax, x0, y0, w, h, edge="#2f3e4e", lw=1.4):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", facecolor="white",
                                edgecolor=edge, linewidth=lw))


def fig5a_schematic(di: pd.DataFrame, n_properties: int, n_components: int, variance: float) -> plt.Figure:
    """v7 Fig. 5a with the physicochemical term drawn as the DI matrix actually used (not BLOSUM62)."""
    with plt.rc_context({"mathtext.fontset": "cm"}):
        return _fig5a(di, n_properties, n_components, variance)


def _fig5a(di, n_properties, n_components, variance):
    W, H = 10.0, 8.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    _box(ax, 0.25, 3.45, 4.35, 4.3)
    ax.text(2.42, 7.38, "Evolutionary constraint", ha="center", fontsize=12, fontweight="bold")
    ax.text(2.42, 6.95, r"EVO score for substitution ($i \rightarrow j$)", ha="center", fontsize=10)
    ax.text(2.42, 5.75, r"$EVO_{\mathrm{score}} = -\log_{10}\left(\dfrac{f_{\mathrm{variant}}}{f_{\mathrm{wildtype}}}\right)$",
            ha="center", va="center", fontsize=15)
    ax.text(0.5, 4.55, r"$f_{\mathrm{variant}}$: frequency of variant at position $i$", fontsize=9)
    ax.text(0.5, 4.15, r"$f_{\mathrm{wildtype}}$: frequency of wild-type residue at position $i$", fontsize=9)
    ax.text(0.5, 3.75, "(ConSurf MSA; scaled as min(EVO, 3) / 3)", fontsize=8, color="#555555")

    _box(ax, 5.0, 3.45, 4.75, 4.3)
    ax.text(7.37, 7.38, "Physicochemical constraint", ha="center", fontsize=12, fontweight="bold")
    ax.text(7.37, 7.0, f"Distance index (DI): {n_properties} AAindex properties", ha="center", fontsize=10)
    ax.text(7.37, 6.72, f"{n_components} principal components ({100 * variance:.1f}% variance)", ha="center", fontsize=9,
            color="#555555")
    inset = fig.add_axes([6.0 / W, 3.62 / H, 2.75 / W, 2.75 / H])
    mat = (di / di.values.max()).loc[AA, AA]
    im = inset.imshow(mat.values, cmap="Blues", vmin=0, vmax=1)
    inset.set_xticks(range(20))
    inset.set_xticklabels(AA, fontsize=6)
    inset.xaxis.tick_top()
    inset.set_yticks(range(20))
    inset.set_yticklabels(AA, fontsize=6)
    inset.tick_params(length=0, pad=1)
    for s in inset.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([8.95 / W, 3.62 / H, 0.12 / W, 2.75 / H])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)

    ax.plot([2.42, 2.42, 7.37, 7.37], [3.4, 3.05, 3.05, 3.4], color="black", lw=1.4)
    ax.add_patch(FancyArrowPatch((3.7, 3.05), (3.7, 2.55), arrowstyle="-|>", mutation_scale=14, color="black", lw=1.4))
    _box(ax, 1.0, 0.45, 5.4, 2.05)
    ax.text(3.7, 2.12, "Combined substitution severity score", ha="center", fontsize=11.5, fontweight="bold",
            color="#4b3f8f")
    ax.text(3.7, 1.3, r"$SSS_{ij} = \dfrac{EVO_{\mathrm{score}}(i,j) + DI(i,j)}{2}$", ha="center", va="center", fontsize=15)
    ax.add_patch(FancyArrowPatch((6.5, 1.47), (7.05, 1.47), arrowstyle="-|>", mutation_scale=16, color="black", lw=2))
    _box(ax, 7.1, 0.45, 2.65, 2.05)
    ax.text(8.42, 2.12, "Severity classification", ha="center", fontsize=11, fontweight="bold")
    for yy, col, name, rule in [(1.45, "#2ca02c", "Tolerated", r"$SSS < 0.5$"), (0.9, "#d62728", "Severe", r"$SSS \geq 0.5$")]:
        ax.add_patch(Circle((7.45, yy), 0.14, facecolor=col, alpha=0.35, edgecolor=col, lw=1.2))
        ax.text(7.7, yy, name, va="center", fontsize=10, fontweight="bold")
        ax.text(8.65, yy, rule, va="center", fontsize=10)
    return fig


def _sss_heatmap(ax, mat: pd.DataFrame, cmap: str, cbar_label: str, vmax: float = 1.0, marks=None):
    ax.set_facecolor("#d9d9d9")
    sns.heatmap(mat, cmap=cmap, vmin=0, vmax=vmax, linewidths=0.2, linecolor="#f2f2f2", xticklabels=True, yticklabels=True,
                cbar_kws={"label": cbar_label, "shrink": 0.8, "aspect": 30}, ax=ax)
    if marks:
        cols = list(mat.columns)
        for col, aa in marks:
            if col in cols and aa in mat.index:
                ax.add_patch(Rectangle((cols.index(col), list(mat.index).index(aa)), 1, 1, fill=False, edgecolor="black", lw=1.0))
    ax.set_ylabel("Substituted Amino Acid", fontsize=8, labelpad=8)
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    ax.figure.axes[-1].yaxis.label.set_fontweight("bold")


def sss_matrix(sss: pd.DataFrame, residues, feat: pd.DataFrame, value: str = "sss") -> pd.DataFrame:
    mat = sss[sss.residue.isin(residues)].pivot(index="mt", columns="residue", values=value).reindex(AA)
    mat = mat[list(residues)]
    mat.columns = [f"{feat.loc[r, 'wt']}{r}" for r in mat.columns]
    return mat


def fig5b_novel_heatmap(sss: pd.DataFrame, novel: pd.DataFrame, feat: pd.DataFrame) -> plt.Figure:
    mat = sss_matrix(sss, sorted(novel.index), feat)
    fig, ax = plt.subplots(figsize=(12, 5))
    _sss_heatmap(ax, mat, "YlOrRd", "Substitution Severity Score (SSS)")
    ax.set_xlabel("Novel Hotspot Residues (Wild-type & Position)", fontsize=8, labelpad=8)
    fig.tight_layout()
    return fig


def clinvar_positions(bench: pd.DataFrame):
    benign = sorted(bench.loc[bench["class"] == "Benign", "residue"].unique())
    patho = sorted(set(bench.loc[bench["class"] == "Pathogenic", "residue"]) - set(benign))
    return benign, patho


def _clinvar_split(ax, n_benign: int, n_patho: int, y: float = -1.0):
    ax.axvline(n_benign, color="black", ls="--", lw=1.5)
    ax.text(n_benign / 2, y, "CLINVAR BENIGN /\nLIKELY BENIGN", ha="center", va="bottom", color="#2ca02c",
            fontweight="bold", fontsize=8)
    ax.text(n_benign + n_patho / 2, y, "CLINVAR PATHOGENIC / LIKELY PATHOGENIC", ha="center", va="bottom",
            color="#d62728", fontweight="bold", fontsize=9)


def fig5c_clinvar_heatmap(sss: pd.DataFrame, bench: pd.DataFrame, feat: pd.DataFrame) -> plt.Figure:
    benign, patho = clinvar_positions(bench)
    mat = sss_matrix(sss, benign + patho, feat)
    marks = [(f"{w}{r}", m) for w, r, m in bench[["wt", "residue", "mt"]].itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(13, 5))
    _sss_heatmap(ax, mat, "YlOrRd", "Substitution Severity Score (SSS)", marks=marks)
    _clinvar_split(ax, len(benign), len(patho))
    ax.set_xlabel("ClinVar-Annotated GAA Missense Positions (Wild-type & Position)", fontsize=8, labelpad=8)
    fig.tight_layout()
    return fig


# =========================================================================================
# Figure 6 — benchmark
# =========================================================================================

def fig6_benchmark(bench: pd.DataFrame, bperf: pd.DataFrame, delong: pd.DataFrame) -> plt.Figure:
    tools = sorted(["SSS", "AlphaMissense", "ESM-1v", "EVE"], key=lambda t: -bperf.loc[t, "roc_auc"])
    y = (bench["class"] == "Pathogenic").astype(int).values
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    for t in tools:
        fpr, tpr, _ = roc_curve(y, bench[t].values)
        col, ls = TOOL_STYLE[t]
        ax.plot(fpr, tpr, color=col, ls=ls, lw=2.0, label=f"{TOOL_LABELS[t]} (AUC = {bperf.loc[t, 'roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="#7f7f7f", lw=1.0, label="Random Chance (AUC = 0.50)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="normal")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight="normal")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"n = {len(bench)} ClinVar variants ({y.sum()} P/LP, {(1 - y).sum()} B/LB)", fontsize=9, fontweight="normal")
    panel_label(ax, "a", x=-0.12, y=1.06)

    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cols = [0.13, 0.40, 0.58, 0.74, 0.90]
    head = ["Model / Predictor", "ROC-AUC\n(95% CI)", "Spearman r", "p-value", "DeLong P\nvs SSS"]
    top, row_h = 0.82, 0.12
    ax.plot([0, 1], [top + 0.09] * 2, color="black", lw=2.0)
    for x, h in zip(cols, head):
        ax.text(x, top + 0.005, h, ha="center", va="center", fontsize=11)
    ax.plot([0, 1], [top - 0.07] * 2, color="black", lw=1.0)
    dp = pd.Series(delong.p.values, index=delong.comparison.str.replace("SSS vs ", "", regex=False))
    for i, t in enumerate(tools):
        r = bperf.loc[t]
        yy = top - 0.07 - row_h * (i + 0.5)
        colr = "#d62728" if t == "SSS" else "black"
        cells = [TOOL_LABELS[t].replace("Combined SSS [This Study]", "SSS (This Study)"),
                 f"{r.roc_auc:.3f}\n({r.bootstrap_ci_low:.3f}–{r.bootstrap_ci_high:.3f})", f"{r.spearman_r:.3f}",
                 sci(r.spearman_p), "—" if t == "SSS" else f"{dp.get(t, np.nan):.2f}"]
        for x, c in zip(cols, cells):
            ax.text(x, yy, c, ha="center", va="center", fontsize=11 if "\n" not in c else 10, color=colr)
    ax.plot([0, 1], [top - 0.07 - row_h * len(tools)] * 2, color="black", lw=2.0)
    ax.text(0, top - 0.07 - row_h * len(tools) - 0.06,
            "ROC-AUC 95% CI: stratified bootstrap (2,000 resamples). Spearman r between score and class (P/LP = 1).",
            fontsize=8, color="#555555", va="top")
    panel_label(ax, "b", x=-0.02, y=0.97)
    return fig


# =========================================================================================
# Supplementary figures
# =========================================================================================

def figS_feature_boxplots(feat: pd.DataFrame, fstats: pd.DataFrame, features: list[str]) -> plt.Figure:
    ncol = 5
    nrow = int(np.ceil(len(features) / ncol))
    with sns.axes_style("whitegrid", {"grid.linestyle": "--"}):
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow))
        for ax, f in zip(axes.ravel(), features):
            df = pd.DataFrame({"label": feat.mutation_observed.map({0: "No Mutation", 1: "Mutation"}), "v": feat[f]})
            sns.boxplot(data=df, x="label", y="v", hue="label", order=["No Mutation", "Mutation"],
                        palette={"No Mutation": LABEL_COLOURS[0], "Mutation": LABEL_COLOURS[1]}, legend=False, ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel(FEATURE_LABELS.get(f, f), fontweight="normal")
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color("#cccccc")
            q = fstats.loc[f, "permutation_q_bh"]
            ax.set_title(stars(q), fontsize=16, fontweight="normal", pad=10)
            x0, x1 = 0.35, 0.65
            ax.plot([x0, x1], [1.03, 1.03], transform=ax.transAxes, color="black", lw=1.5, clip_on=False)
        for ax in axes.ravel()[len(features):]:
            ax.axis("off")
        fig.tight_layout(h_pad=2.0)
    return fig


def fold_metrics(long: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    rows = []
    for (rep, fold, m), d in long.groupby(["repeat", "fold", "model"]):
        pred = (d.probability >= threshold).astype(int)
        rows.append({"repeat": rep, "fold": fold, "model": m, "accuracy": accuracy_score(d.y, pred),
                     "f1": f1_score(d.y, pred), "roc_auc": roc_auc_score(d.y, d.probability),
                     "balanced_accuracy": balanced_accuracy_score(d.y, pred)})
    return pd.DataFrame(rows)


def figS_model_metrics(fm: pd.DataFrame) -> plt.Figure:
    panels = [("accuracy", "Accuracy by Model", "Accuracy"), ("f1", "F1 Score by Model", "F1 Score"),
              ("roc_auc", "AUC Scores by Model", "AUC Score"), ("balanced_accuracy", "Balanced Accuracy by Model", "Balanced Accuracy")]
    with sns.axes_style("whitegrid", {"grid.linestyle": "--"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
        for ax, (col, title, ylab), letter in zip(axes.ravel(), panels, "ABCD"):
            s = fm.groupby("model")[col].agg(["mean", "std"]).loc[BAR_MODEL_ORDER]
            ax.bar(range(5), s["mean"], yerr=s["std"], color=[BAR_COLOURS[m] for m in BAR_MODEL_ORDER],
                   edgecolor="black", linewidth=1.2, capsize=5, error_kw=dict(lw=1.5))
            for i, (mu, sd) in enumerate(zip(s["mean"], s["std"])):
                ax.text(i, mu + sd + 0.015, f"{mu:.2f}", ha="center", fontsize=8)
            ax.set_xticks(range(5))
            ax.set_xticklabels([MODEL_LABELS[m].replace("Support Vector Machine", "SVM") for m in BAR_MODEL_ORDER],
                               rotation=45, ha="right")
            ax.set_title(title)
            ax.set_ylabel(ylab, fontweight="normal")
            ax.set_ylim(0, max(1.0, (s["mean"] + s["std"]).max() + 0.08))
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color("#cccccc")
            ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
        fig.tight_layout()
    return fig


def figS_confusion(prob: pd.DataFrame, thresholds: dict) -> plt.Figure:
    y = prob["mutation_observed"].values
    rows = [("Probability ≥ 0.5", {m: 0.5 for m in V7_MODEL_ORDER}),
            ("90% specificity vote threshold", thresholds)]
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = ["No mutation", "Mutation"]
    for r, (rule, thr) in enumerate(rows):
        for c, m in enumerate(V7_MODEL_ORDER):
            ax = axes[r, c]
            cm = confusion_matrix(y, (prob[m].values >= thr[m]).astype(int), labels=[0, 1])
            pct = 100 * cm / cm.sum()
            annot = np.array([[f"{cm[i, j]}\n({pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)])
            sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=len(y) - y.sum(), cbar=False, xticklabels=names, yticklabels=names, square=True,
                        annot_kws={"fontsize": 12}, ax=ax)
            sens, spec = cm[1, 1] / cm[1].sum(), cm[0, 0] / cm[0].sum()
            ax.set_title(f"{MODEL_SHORT[m]} Confusion Matrix (Nested CV)\nthreshold {thr[m]:.3f}; "
                         f"sensitivity {sens:.2f}, specificity {spec:.2f}", fontsize=10)
            ax.set_xlabel("Predicted Label", fontweight="normal")
            ax.set_ylabel("True Label" if c == 0 else "", fontweight="normal")
        axes[r, 0].text(-0.55, 0.5, rule, transform=axes[r, 0].transAxes, rotation=90, ha="center", va="center",
                        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, len(y) - y.sum()))
    fig.colorbar(sm, cax=fig.add_axes([0.96, 0.15, 0.01, 0.7]), label="Number of Residues")
    return fig


def figS_cv_roc(long: pd.DataFrame) -> plt.Figure:
    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.grid(True, color="#e5e5e5", linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)
    for m in V7_MODEL_ORDER:
        tprs, aucs = [], []
        for _, d in long[long.model == m].groupby(["repeat", "fold"]):
            fpr, tpr, _ = roc_curve(d.y, d.probability)
            t = np.interp(grid, fpr, tpr)
            t[0] = 0.0
            tprs.append(t)
            aucs.append(auc(fpr, tpr))
        mu, sd = np.mean(tprs, 0), np.std(tprs, 0)
        mu[-1] = 1.0
        ax.plot(grid, mu, color=MODEL_COLOURS[m], lw=1.3,
                label=f"{MODEL_LABELS[m]} (AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f})")
        ax.fill_between(grid, np.maximum(mu - sd, 0), np.minimum(mu + sd, 1), color=MODEL_COLOURS[m], alpha=0.10)
    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1.0, label="Random baseline")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("False Positive Rate", labelpad=8)
    ax.set_ylabel("True Positive Rate", labelpad=8)
    ax.set_title("Cross-Validated ROC Curves (5 × 10-fold nested CV)", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#eaeaf2", fontsize=7.5)
    fig.tight_layout()
    return fig


def figS_permutation_importance(pi: pd.DataFrame) -> plt.Figure:
    pi = pi.sort_values("mean_across_models")
    fig, ax = plt.subplots(figsize=(7.5, 7))
    tidy(ax, grid="x")
    h = 0.16
    for k, m in enumerate(V7_MODEL_ORDER):
        ax.barh(np.arange(len(pi)) + (k - 2) * h, pi[m], height=h, color=MODEL_COLOURS[m], label=MODEL_LABELS[m])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(pi)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f in pi.index])
    ax.set_xlabel("Decrease in held-out ROC AUC when permuted", labelpad=8)
    ax.set_title("Permutation Feature Importance (outer CV folds)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def figS_shap(sv: pd.DataFrame, xin: pd.DataFrame, title: str) -> plt.Figure:
    import shap
    shap.summary_plot(sv.values, features=xin.values, feature_names=[FEATURE_LABELS[c] for c in xin.columns],
                      show=False, plot_size=(7.5, 6.5))
    fig = plt.gcf()
    plt.title(title, fontsize=11, fontweight="bold")
    return fig


def figS_predicted_vs_clinical(votes: pd.DataFrame, min_votes: int) -> plt.Figure:
    pred = set(votes.index[votes.votes >= min_votes])
    known = set(votes.index[votes.mutation_observed == 1])
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), gridspec_kw={"width_ratios": [1.1, 0.9, 1.1]})
    ax = axes[0]
    v = venn2([pred, known], set_labels=(f"Model Predicted Hotspots\n(≥{min_votes}/5 votes; {len(pred)})",
                                          f"Clinical GAA Mutation Positions\n({len(known)})"), ax=ax)
    v.get_patch_by_id("10").set_color("#d62728")
    v.get_patch_by_id("01").set_color("#1f77b4")
    v.get_patch_by_id("11").set_color("#9467bd")
    for rid in ("10", "01", "11"):
        v.get_patch_by_id(rid).set_alpha(0.45)
        v.get_label_by_id(rid).set_fontsize(12)
        v.get_label_by_id(rid).set_fontweight("bold")
    venn2_circles([pred, known], linewidth=0.8, color="#555555", ax=ax)
    ax.set_title("Venn Diagram of Residue Comparison", fontsize=12)
    panel_label(ax, "a", x=-0.05, y=1.0)

    ax = axes[1]
    common = len(pred & known)
    ax.bar(1, len(known), color="lightblue", width=0.5, label="Clinical mutation positions")
    ax.bar(1, common, color="blue", width=0.5, label="Common residues")
    ax.bar(2, len(pred), color="lightcoral", width=0.5, label="Model predicted hotspots")
    ax.bar(2, common, color="red", width=0.5, label="Common residues ")
    for x, tot in [(1, len(known)), (2, len(pred))]:
        ax.text(x, tot + 3, str(tot), ha="center", fontsize=10)
        ax.text(x, common / 2, str(common), ha="center", color="white", fontsize=10, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["GAA Mutation List", "Model Prediction List"])
    ax.set_xlabel("Residue Categories", labelpad=8)
    ax.set_ylabel("Count")
    ax.set_title("GAA Mutation List vs Model Prediction List", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    tidy(ax, grid="y")
    panel_label(ax, "b", x=-0.15, y=1.0)

    ax = axes[2]
    vc = votes.groupby(["votes", "mutation_observed"]).size().unstack(fill_value=0).reindex(range(6), fill_value=0)
    ax.bar(vc.index - 0.2, vc.get(0, 0), width=0.4, color="#1f77b4", edgecolor="black", label="No Mutation")
    ax.bar(vc.index + 0.2, vc.get(1, 0), width=0.4, color="#ff7f0e", edgecolor="black", label="Mutation")
    for xi in vc.index:
        for off, k in [(-0.2, 0), (0.2, 1)]:
            ax.text(xi + off, vc.loc[xi].get(k, 0) + 5, str(vc.loc[xi].get(k, 0)), ha="center", fontsize=8)
    ax.axvspan(min_votes - 0.5, 5.5, color="#ffe9a8", alpha=0.5, lw=0, zorder=0)
    ax.text(4, ax.get_ylim()[1] * 0.6, f"Consensus\n(≥{min_votes} of 5)", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(6))
    ax.set_xlabel("Number of Models Voting (pred_all)", labelpad=8)
    ax.set_ylabel("Residues")
    ax.set_title("Distribution of Model Votes", fontsize=12)
    ax.legend(loc="upper right")
    tidy(ax, grid="y")
    panel_label(ax, "c", x=-0.15, y=1.0)
    fig.tight_layout()
    return fig


def figS_hotspot_stability(novel: pd.DataFrame) -> plt.Figure:
    cm = {3: "#fdae6b", 4: "#fd8d3c", 5: "#d94801"}
    nv = novel.sort_values(["consensus_repeat_fraction", "votes"], ascending=False)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    tidy(ax, grid="y")
    ax.bar(range(len(nv)), nv.consensus_repeat_fraction, color=[cm[int(v)] for v in nv.votes], edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(nv)))
    ax.set_xticklabels(nv.residue_label, rotation=90, fontsize=8)
    ax.set_ylabel("Fraction of CV repeats\nreaching consensus")
    ax.set_xlabel("Novel Consensus Hotspots", labelpad=8)
    ax.set_ylim(0, 1.05)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="black", label=f"{k}/5 votes") for k, c in cm.items()],
              loc="upper right", ncol=3)
    fig.tight_layout()
    return fig


def figS_di_evoscore(di: pd.DataFrame, sss: pd.DataFrame, cap: float, title: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    sns.heatmap((di / di.values.max()).loc[AA, AA], cmap="Greens", annot=True, fmt=".2f", annot_kws={"fontsize": 5.5},
                square=True, cbar_kws={"label": "Physicochemical Distance (DI, scaled)", "shrink": 0.8}, ax=ax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Substituted Amino Acid", labelpad=6)
    ax.set_ylabel("Wild-type Amino Acid", labelpad=6)
    ax.tick_params(axis="y", labelrotation=0)
    panel_label(ax, "a", x=-0.1, y=1.02)
    ax = axes[1]
    tidy(ax)
    sns.histplot(sss.evoscore, bins=60, kde=True, color="#9467bd", ax=ax)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(cap, color="#d62728", ls="--", lw=1.2, label=f"cap = {cap:g} (scaled EVOscore = 1)")
    ax.set_xlabel(r"EVOscore = $-\log_{10}(f_{variant} / f_{wildtype})$", labelpad=6)
    ax.set_ylabel("Number of Substitutions")
    ax.set_title(f"EVOscore of all {len(sss):,} possible substitutions", fontsize=11)
    ax.legend()
    panel_label(ax, "b", x=-0.1, y=1.02)
    fig.tight_layout()
    return fig


def figS_full_protein_sss(sss: pd.DataFrame) -> plt.Figure:
    full = sss.pivot(index="mt", columns="residue", values="sss").reindex(AA)
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.set_facecolor("#d9d9d9")
    sns.heatmap(full, cmap="YlOrRd", vmin=0, vmax=1, xticklabels=50,
                cbar_kws={"label": "Substitution Severity Score (SSS)", "shrink": 0.8, "aspect": 30}, ax=ax)
    ax.set_xlabel("Residue Number", labelpad=8)
    ax.set_ylabel("Substituted Amino Acid", labelpad=8)
    ax.tick_params(axis="y", labelrotation=0)
    ax.set_title("Substitution Severity Score for every possible substitution (residues 81–952)", fontsize=11)
    fig.tight_layout()
    return fig


def figS_clinvar_components(sss: pd.DataFrame, bench: pd.DataFrame, feat: pd.DataFrame) -> plt.Figure:
    benign, patho = clinvar_positions(bench)
    pos = benign + patho
    marks = [(f"{w}{r}", m) for w, r, m in bench[["wt", "residue", "mt"]].itertuples(index=False)]
    panels = [("evoscore_scaled", "Purples", "Evolutionary Constraint (EVO, scaled)", "Sequence Evolutionary Constraint"),
              ("di_scaled", "Greens", "Physicochemical Distance (DI, scaled)", "Physicochemical Distance"),
              ("sss", "YlOrRd", "Combined Severity Score (SSS)", "Combined Severity (SSS = (EVO + DI) / 2)")]
    fig, axes = plt.subplots(3, 1, figsize=(16, 16))
    for ax, (col, cmap, lab, title), letter in zip(axes, panels, "abc"):
        _sss_heatmap(ax, sss_matrix(sss, pos, feat, col), cmap, lab, marks=marks)
        _clinvar_split(ax, len(benign), len(patho), y=-0.6)
        ax.set_title(title, pad=30)
        ax.set_xlabel("Positions (Benign | Pathogenic)", fontsize=8)
        panel_label(ax, letter, x=-0.05, y=1.08)
    fig.tight_layout(h_pad=3)
    return fig


def figS_benchmark_separation(bench: pd.DataFrame, bperf: pd.DataFrame) -> plt.Figure:
    tools = ["SSS", "AlphaMissense", "ESM-1v", "EVE", "EVOscore", "DI"]
    tools = sorted(tools, key=lambda t: -bperf.loc[t, "roc_auc"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    ax = axes[0]
    tidy(ax, grid="y")
    r = bperf.loc[tools]
    ax.bar(range(len(tools)), r.roc_auc, yerr=[r.roc_auc - r.bootstrap_ci_low, r.bootstrap_ci_high - r.roc_auc],
           color=[TOOL_STYLE[t][0] for t in tools], edgecolor="black", width=0.55, capsize=4)
    for i, v in enumerate(r.roc_auc):
        ax.text(i, r.bootstrap_ci_high.iloc[i] + 0.01, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(tools)))
    ax.set_xticklabels([TOOL_LABELS[t].replace(" [This Study]", "") for t in tools], rotation=20, ha="right")
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("ROC-AUC Score (95% bootstrap CI)")
    ax.set_title("AUC Performance Ranking")
    panel_label(ax, "a", x=-0.12, y=1.04)

    ax = axes[1]
    tidy(ax, grid="y")
    main = [t for t in tools if t in ("SSS", "AlphaMissense", "ESM-1v", "EVE")]
    y = bench["class"].values
    for i, t in enumerate(main):
        s = bench[t]
        s = (s - s.min()) / (s.max() - s.min())
        for k, off in [("Benign", -0.22), ("Pathogenic", 0.22)]:
            ax.boxplot([s[y == k].values], positions=[i + off], widths=0.36, patch_artist=True, showfliers=True,
                       flierprops=dict(marker="o", markersize=2.5, alpha=0.4, markerfacecolor=CLASS_COLOURS[k], linestyle="none"),
                       medianprops=dict(color="black", lw=1.2), boxprops=dict(facecolor=CLASS_COLOURS[k], alpha=0.6, lw=0.8),
                       whiskerprops=dict(lw=0.8, ls="--"), capprops=dict(lw=0.8))
        ax.plot([i - 0.22, i + 0.22], [1.05, 1.05], "k-", lw=0.8)
        ax.text(i, 1.06, stars(bperf.loc[t, "mann_whitney_p"]), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(main)))
    ax.set_xticklabels([TOOL_LABELS[t].replace(" [This Study]", "") for t in main], fontweight="bold")
    ax.set_ylim(-0.05, 1.18)
    ax.set_ylabel("Normalized Score Value (0 to 1)")
    ax.set_title("Score Separation (Benign vs. Pathogenic)")
    nb, npth = (y == "Benign").sum(), (y == "Pathogenic").sum()
    ax.legend(handles=[Patch(facecolor=CLASS_COLOURS["Benign"], alpha=0.6, edgecolor="black", label=f"Benign (n={nb})"),
                       Patch(facecolor=CLASS_COLOURS["Pathogenic"], alpha=0.6, edgecolor="black", label=f"Pathogenic (n={npth})")],
              loc="lower right", frameon=True)
    panel_label(ax, "b", x=-0.1, y=1.04)
    fig.tight_layout()
    return fig


def figS_legacy811(perf: pd.DataFrame, legacy: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    tidy(ax, grid="y")
    xs = np.arange(5)
    for off, src, lab, col in [(-0.2, perf, "All 844 residues (this analysis)", "#1f77b4"),
                               (0.2, legacy, "Original 811-residue subset", "#ff7f0e")]:
        s = src.loc[V7_MODEL_ORDER]
        ax.bar(xs + off, s.pooled_oof_auc, width=0.4, color=col, edgecolor="black", label=lab,
               yerr=[s.pooled_oof_auc - s.auc_ci_low, s.auc_ci_high - s.pooled_oof_auc], capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS[m].replace("Support Vector Machine", "SVM") for m in V7_MODEL_ORDER], rotation=20, ha="right")
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("Pooled out-of-fold ROC AUC (95% CI)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def figS_ablation_vote_rule(ab: pd.DataFrame, ts: pd.DataFrame) -> plt.Figure:
    sets = list(dict.fromkeys(ab.feature_set))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    tidy(ax, grid="x")
    for off, m in [(-0.15, "lr"), (0.15, "xgb")]:
        d = ab[ab.model == m].set_index("feature_set").loc[sets]
        ax.errorbar(d.pooled_oof_auc, np.arange(len(sets)) + off, xerr=[d.pooled_oof_auc - d.auc_ci_low, d.auc_ci_high - d.pooled_oof_auc],
                    fmt="o", ms=6, color=MODEL_COLOURS[m], elinewidth=1.2, capsize=3, label=MODEL_LABELS[m])
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels(sets)
    ax.invert_yaxis()
    ax.set_xlabel("Pooled out-of-fold ROC AUC (95% CI)", labelpad=8)
    ax.set_title("Feature-Group Ablation")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    panel_label(ax, "a", x=-0.42, y=1.04)
    ax = axes[1]
    tidy(ax, grid="y")
    xs = np.arange(len(ts))
    ax.bar(xs - 0.2, ts.consensus_known_positions, width=0.4, color="#ff7f0e", edgecolor="black", label="Known mutation positions")
    ax.bar(xs + 0.2, ts.novel_consensus_hotspots, width=0.4, color="#1f77b4", edgecolor="black", label="Novel hotspots")
    for x, v in zip(xs, ts.novel_consensus_hotspots):
        ax.text(x + 0.2, v + 3, str(v), ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(ts.vote_rule, rotation=30, ha="right")
    ax.set_ylabel("Consensus residues (≥3 of 5 votes)")
    ax.set_title("Sensitivity to the Vote Rule")
    ax.legend(loc="upper right")
    panel_label(ax, "b", x=-0.12, y=1.04)
    fig.tight_layout()
    return fig
