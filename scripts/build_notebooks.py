"""Write and execute the step-by-step analysis notebooks in notebooks/ (run after run_all.sh).

The notebooks follow the structure of the original analysis notebooks (full_rin_analysis,
classification_model): every step is shown, from the raw input files to the figures.
Each notebook recomputes its numbers from data/raw and checks them against the pipeline
outputs in data/processed and results/tables. Figures are drawn with the same functions
as scripts/08_figures.py, so they are identical to the files in results/figures/.
"""

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)


def md(s):
    return ("md", s)


def code(s):
    return ("code", s)


SETUP = code("""%matplotlib inline
import sys, json, inspect, time, warnings
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from gaa_hotspot import data, network, stats, ml, figures as F
from gaa_hotspot import sss as sss_mod
from gaa_hotspot.common import AA, MODEL_ORDER, MODEL_NAMES, config, raw, read_processed, table, load_values
from gaa_hotspot.plotting import apply_style, MODEL_COLOURS, MODEL_LABELS, MODEL_SHORT, V7_MODEL_ORDER

apply_style()
plt.rcParams["figure.dpi"] = 80
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 4)
cfg = config()
V = load_values()
print("project root:", ROOT)""")

# =============================================================================================
# 01  RIN criticality analysis
# =============================================================================================
NB01 = [
    md("""# 01 · Residue interaction network (RIN) and criticality analysis

Human lysosomal acid α-glucosidase (GAA), PDB 5NN3 (Rosetta-relaxed), chain A.

This notebook rebuilds every network quantity from the raw RINerator output and follows the steps of the original `full_rin_analysis` notebook:

1. Load the RINerator network and build the graph
2. Compute the graph features (centralities)
3. Check the residues with zero betweenness and the "disconnected" residues removed in earlier drafts
4. Add the structural (Chimera) and conservation (ConSurf-DB) features and the mutation labels
5. Correlation between the graph features
6. Criticality groups (LDLB / LDHB / HDHB) by K-means binning → **Fig. 3a**
7. Supercritical hubs by the Euclidean distance to the ideal point → **Fig. 3c**
8. Mutation positions in the groups → **Fig. 3e**
9. Solvent exposure and conservation of the groups → **Fig. 4c, d**
10. Mutation vs no-mutation feature comparison → **Fig. S1, Table S2**
11. Reproduction of the earlier 811-residue analysis"""),
    SETUP,
    md("## 1. Object for analysis\n\nRaw input files used in this notebook:"),
    code("""prot = "5NN3_relaxed"
for sub in ["structure", "rin", "chimera", "conservation", "labels"]:
    for f in sorted(raw(sub).glob("*")):
        print(f"{sub:13s} {f.name:42s} {f.stat().st_size / 1024:9.1f} kB")"""),
    md("## 2. RINerator network\n\nThe SIF file lists one typed interaction per line (`A:<residue>:_:<aa>  <type>  A:<residue>:_:<aa>`)."),
    code("""sif = raw("rin", f"{prot}_h.sif").read_text().splitlines()
print(len(sif), "lines in the SIF file\\n")
print(*sif[:8], sep="\\n")"""),
    code("""typed = data.rin_edges()          # (u, v, interaction type); self-pairs removed
print("typed interactions:", len(typed))
typed["type"].value_counts().to_frame("count")"""),
    md("A residue pair can be joined by several interaction types (e.g. a side-chain contact and a hydrogen bond). The graph uses each **distinct residue pair once**:"),
    code("""pairs = typed[["u", "v"]].drop_duplicates()
multi = (typed.groupby(["u", "v"]).size() > 1).sum()
print("unique residue-residue contacts:", len(pairs))
print("pairs joined by more than one interaction type:", int(multi))"""),
    code("""g = network.build_graph(typed)
comps = g.components()
nodes = sorted(int(n) for n in g.vs["name"])
gaps = sorted(set(range(nodes[0], nodes[-1] + 1)) - set(nodes))
print(f"nodes: {g.vcount()}   edges: {g.ecount()}   connected components: {len(comps)}   connected: {g.is_connected()}")
print(f"residue range: {nodes[0]}-{nodes[-1]}")
print(f"{len(gaps)} residue numbers inside this range are not in the model (unresolved in the crystal structure):")
print(gaps)"""),
    md("""## 3. Graph features of the RIN

| feature | meaning |
|---|---|
| degree | number of residues in contact |
| betweenness | number of shortest paths passing through the residue |
| closeness | inverse mean shortest-path length to all other residues |
| burt | Burt's constraint (low = bridges otherwise separate neighbourhoods) |
| authority | HITS authority score |
| page_rank | PageRank |
| k_core | largest k such that the residue is in the k-core |"""),
    code("""df_graph_features = network.centralities(g)
print(df_graph_features.shape)
df_graph_features.head()"""),
    code("df_graph_features.describe().T"),
    md("""## 4. Important preprocessing: zero betweenness

The earlier notebook removed residues with zero betweenness before taking log10(betweenness), because log10(0) = −∞.
Here all residues are kept and the feature is **log10(1 + betweenness)**, which is defined for every residue and identical in shape for the others."""),
    code("""zero_betweenness = df_graph_features.query("betweenness == 0").index.tolist()
original_zero = [97, 114, 155, 187, 189, 206, 226, 288, 418, 419, 508, 531, 678, 686, 799, 849, 850, 882, 894, 912, 914, 924, 931, 940, 946]
print(len(zero_betweenness), "residues with zero betweenness:", zero_betweenness)
print("same list as the earlier notebook:", zero_betweenness == original_zero)
df_graph_features.loc[zero_betweenness, ["degree", "closeness", "k_core"]].T"""),
    md("""## 5. The "disconnected" residues of the earlier notebook

The earlier notebook also removed a hard-coded list of 16 residues described as *disconnected*. The graph above is a single connected component, so every residue can be reached from every other one. The table shows each listed residue with its neighbours:"""),
    code("""old_disconnected = [97, 206, 337, 288, 88, 831, 483, 651, 758, 814, 678, 799, 848, 946, 914, 924]
seq = data.structure_sequence()
giant = {int(g.vs[i]["name"]) for i in max(comps, key=len)}
rows = []
for r in sorted(old_disconnected):
    v = g.vs.find(name=str(r))
    rows.append({"residue": f"{seq[r]}{r}", "degree": v.degree(),
                 "neighbours": ", ".join(sorted((g.vs[n]["name"] for n in g.neighbors(v)), key=int)),
                 "betweenness": round(df_graph_features.loc[r, "betweenness"], 1),
                 "in_main_component": r in giant,
                 "mean_shortest_path": round(float(np.mean(g.distances(source=v)[0])), 2)})
pd.DataFrame(rows)"""),
    code("""labels = data.mutation_labels()
extra = [r for r in old_disconnected if r not in zero_betweenness]
print("listed as disconnected although betweenness > 0:", extra)
print("  of these, reported mutation positions:", [f"{seq[r]}{r}" for r in extra if labels.get(r, 0) == 1])
print("zero-betweenness residues that are mutation positions:", [f"{seq[r]}{r}" for r in zero_betweenness if labels.get(r, 0) == 1])
print(f"residues removed in the earlier analysis: {len(set(zero_betweenness) | set(old_disconnected))}  ->  844 - 33 = 811")"""),
    md("## 6. log betweenness\n\nAdded to reduce the skew of betweenness:"),
    code("""assert np.allclose(np.log10(1 + df_graph_features.betweenness), df_graph_features.log_betweenness)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
sns.histplot(df_graph_features.betweenness, bins=50, ax=axes[0]); axes[0].set_title("betweenness")
sns.histplot(df_graph_features.log_betweenness, bins=50, ax=axes[1]); axes[1].set_title("log10(1 + betweenness)")
plt.tight_layout(); plt.show()"""),
    md("## 7. Structural features (UCSF Chimera)\n\nChimera `save attribute` files, one residue per line:"),
    code("""print(*raw("chimera", "areaSAS.txt").read_text().splitlines()[:6], sep="\\n")
df_structure = data.chimera_features()
print("\\n", df_structure.shape)
display(df_structure.head())
print("missing values:", df_structure.isnull().sum().to_dict())"""),
    md("φ is undefined for the first residue and ψ for residues before a chain break, so a few values are missing. They are imputed with the training-fold median inside the cross-validation (notebook 02)."),
    md("## 8. Conservation (ConSurf-DB)\n\nNormalised ConSurf score per residue; **lower = more conserved**."),
    code("""df_conserv = data.consurf_db_scores()
print(len(df_conserv), "residues with a ConSurf-DB score")
df_conserv.describe().to_frame().T"""),
    md("## 9. Mutation labels\n\n`mutation_observed = 1` for residue positions with a reported Pompe-disease missense mutation (HGMD-derived position list)."),
    code("""df_mutation = data.mutation_labels()
df_mutation.value_counts().rename({0: "no mutation", 1: "mutation position"})"""),
    md("## 10. Making the full dataset\n\nMerge graph, structural and conservation features and the labels; check that the result equals the pipeline file `data/processed/residue_features.csv`."),
    code("""df_full = df_graph_features.join(df_structure, how="left").join(df_conserv, how="left").join(df_mutation, how="left")
df_full.insert(0, "wt", seq.reindex(df_full.index))
df_full["mutation_observed"] = df_full.mutation_observed.fillna(0).astype(int)
stored = read_processed("residue_features.csv").set_index("residue")
num = stored.select_dtypes("number").columns
print(df_full.shape)
print("identical to data/processed/residue_features.csv:",
      np.allclose(df_full[num].astype(float), stored[num].astype(float), equal_nan=True))
df_full.head()"""),
    code("""print(df_full.mutation_observed.value_counts(normalize=True).mul(100).round(2))
sns.countplot(x="mutation_observed", data=df_full, hue="mutation_observed", legend=False)
plt.title("Distribution of Target Values"); plt.xlabel("Target Value"); plt.ylabel("Count"); plt.show()"""),
    md("## 11. Graph feature correlations\n\nAbsolute correlation between the seven centralities (lower triangle), for the three correlation methods used in the original notebook. Burt's constraint is negatively correlated with the others; the sign is in Supplementary Table S3."),
    code("""graph_features = ["betweenness", "page_rank", "degree", "burt", "k_core", "closeness", "authority"]
fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
for ax, method in zip(axes, ["pearson", "kendall", "spearman"]):
    c = df_full[graph_features].corr(method=method)
    mask = np.triu(np.ones_like(c, dtype=bool)); np.fill_diagonal(mask, False)
    sns.heatmap(c.abs(), annot=True, fmt=".2f", vmin=0, vmax=1, square=True, cmap="Reds", mask=mask, ax=ax)
    ax.set_title(f"Correlation method: {method}")
plt.tight_layout(); plt.show()"""),
    md("## 12. Critical residue analysis (criticality)\n\nDegree against log betweenness; the LDLB, LDHB and HDHB regions are visible as corners of the distribution."),
    code("""sns.scatterplot(data=df_full, x="degree", y="log_betweenness")
plt.show()"""),
    md("K-means binning (4 bins) of log betweenness and degree, exactly as `analysis_utils.find_criticality_intervals(..., type_bins='kmeans')` in the original code:"),
    code("""from sklearn.preprocessing import KBinsDiscretizer
kbd = KBinsDiscretizer(n_bins=cfg["criticality"]["kbins"], encode="ordinal", strategy="kmeans", random_state=cfg["seed"])
bins = kbd.fit_transform(df_full[["log_betweenness", "degree"]])
lb_edges, deg_edges = np.sort(kbd.bin_edges_[0]), np.sort(kbd.bin_edges_[1])
lb_intervals = [pd.Interval(a, b, closed="both") for a, b in zip(lb_edges[:-1], lb_edges[1:])]
degree_intervals = [pd.Interval(a, b, closed="both") for a, b in zip(deg_edges[:-1], deg_edges[1:])]
print("log betweenness intervals:"); display(lb_intervals)
print("degree intervals:"); display(degree_intervals)"""),
    md("""Classification rule (`classify_critical`):
- **LDLB**: lowest degree bin *and* lowest log-betweenness bin
- **HDHB**: highest degree bin *and* highest log-betweenness bin
- **LDHB**: lowest degree bin *and* highest log-betweenness bin
- everything else: None"""),
    code("""lb_bin, deg_bin = bins[:, 0], bins[:, 1]
top = cfg["criticality"]["kbins"] - 1
df_full["criticality"] = np.select([(deg_bin == 0) & (lb_bin == 0), (deg_bin == top) & (lb_bin == top), (deg_bin == 0) & (lb_bin == top)],
                                   ["LDLB", "HDHB", "LDHB"], "None")
groups = read_processed("residue_network_groups.csv").set_index("residue")
print("identical to the pipeline:", bool((groups.criticality.replace({"Other": "None"}) == df_full.criticality).all()))
df_full.criticality.value_counts()"""),
    md("### Figure 3a — criticality quadrants"),
    code("""F.fig3a_criticality(df_full, groups, {"log_betweenness": lb_edges, "degree": deg_edges})
plt.show()"""),
    code("""for gname in ["LDLB", "LDHB", "HDHB"]:
    res = df_full.index[df_full.criticality == gname]
    print(f"{gname} ({len(res)}):", ", ".join(f"{df_full.wt[r]}{r}" for r in res), "\\n")"""),
    md("## 13. Supercritical residues through the Euclidean distance method\n\nThe ideal point is the maximum of degree, log betweenness and closeness; residues closest to it are the best trade-off (supercritical) hubs."),
    code("""from scipy.spatial.distance import cdist
pts = df_full[["degree", "log_betweenness", "closeness"]]
ideal_point = pts.max().values
print("ideal point (degree, log betweenness, closeness):", ideal_point.round(4))
df_full["distance_from_ideal"] = cdist(pts.values, ideal_point.reshape(1, -1)).ravel()
best_trade_off_points = df_full.sort_values("distance_from_ideal").head(cfg["criticality"]["n_supercritical"])
best_trade_off_points[["wt", "degree", "log_betweenness", "closeness", "distance_from_ideal", "criticality", "mutation_observed"]]"""),
    code("""sc_list = best_trade_off_points.index.tolist()
print("SUPERCRITICAL RESIDUES:", ", ".join(f"{df_full.wt[r]}{r}" for r in sc_list))
print("same 10 residues as the earlier notebook:", sorted(sc_list) == sorted([292, 350, 710, 713, 391, 499, 575, 498, 661, 660]))
print("identical to the pipeline:", set(sc_list) == set(groups.index[groups.supercritical]))
print("all in HDHB:", bool((best_trade_off_points.criticality == "HDHB").all()))
print("reported mutation positions among them:", int(best_trade_off_points.mutation_observed.sum()), "of", len(sc_list),
      "->", [f"{df_full.wt[r]}{r}" for r in sc_list if df_full.mutation_observed[r] == 1])"""),
    md("### Figure 3c — 3D topological space and distance to the ideal hub"),
    code("""F.fig3c_supercritical_3d(df_full, groups)
plt.show()"""),
    md("**Sensitivity check.** The three axes have very different ranges, so the unscaled distance is dominated by degree. Repeating the selection on z-scored axes:"),
    code("""print("axis ranges  degree:", np.ptp(pts.degree), "  log betweenness:", round(np.ptp(pts.log_betweenness), 2), "  closeness:", round(np.ptp(pts.closeness), 3))
z = (pts - pts.mean()) / pts.std(ddof=0)
dz = pd.Series(cdist(z.values, z.max().values.reshape(1, -1)).ravel(), index=pts.index)
sc_z = dz.nsmallest(10).index
print("z-scored hubs:", ", ".join(f"{df_full.wt[r]}{r}" for r in sc_z))
print("shared with the unscaled set:", len(set(sc_z) & set(sc_list)), "of 10")"""),
    md("## 14. Mutation positions in the criticality groups"),
    code("""from scipy.stats import fisher_exact
rows = []
for gname in ["HDHB", "LDHB", "LDLB", "None"]:
    sel = df_full.criticality == gname
    rows.append({"group": gname, "n": int(sel.sum()), "mutation_positions": int(df_full.mutation_observed[sel].sum()),
                 "fraction_mutated": df_full.mutation_observed[sel].mean()})
rows.append({"group": "SC (subset of HDHB)", "n": len(sc_list), "mutation_positions": int(best_trade_off_points.mutation_observed.sum()),
             "fraction_mutated": best_trade_off_points.mutation_observed.mean()})
display(pd.DataFrame(rows))
res = fisher_exact(pd.crosstab(df_full.criticality == "HDHB", df_full.mutation_observed).values)
print(f"HDHB vs all other residues, odds ratio of being a mutation position = {res.statistic:.2f}, Fisher exact P = {res.pvalue:.2e}")
print(f"overall fraction of mutation positions: {df_full.mutation_observed.mean():.3f}")"""),
    md("### Figure 3e — overlap of mutation positions, HDHB residues and supercritical hubs"),
    code("""F.fig3e_venn(df_full, groups)
plt.show()"""),
    md("## 15. areaSAS and conservation score vs criticality\n\nFirst the quick strip plots of the original notebook, then the manuscript panels (Fig. 4c, d), where the 10 supercritical hubs are shown as their own group (SC) and removed from HDHB."),
    code("""plot_df = df_full.assign(criticality=F.plot_groups(groups))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col in zip(axes, ["areaSAS", "conservation"]):
    sns.stripplot(data=plot_df.query("criticality != 'None'"), x="criticality", y=col, hue="criticality",
                  edgecolor="black", alpha=.7, size=5, linewidth=1.0, jitter=0.05, order="HDHB LDHB LDLB SC".split(),
                  palette="deep", legend=False, ax=ax)
plt.tight_layout(); plt.show()"""),
    code("""from scipy.stats import kruskal
for col in ["areaSAS", "conservation"]:
    med = plot_df.query("criticality != 'None'").groupby("criticality")[col].median().round(3).to_dict()
    p = kruskal(*[df_full.loc[df_full.criticality == g, col] for g in ["HDHB", "LDHB", "LDLB"]]).pvalue
    print(f"{col:13s} medians {med}   Kruskal-Wallis (HDHB, LDHB, LDLB) P = {p:.2e}")"""),
    code("""F.fig4_group_profile(df_full, groups, "areaSAS", "Solvent Accessible Surface Area (SASA, Å²)")
plt.show()"""),
    code("""F.fig4_group_profile(df_full, groups, "conservation", "ConSurf Conservation Score (SCORE)")
plt.show()"""),
    md("## 16. Difference of feature means between mutated and non-mutated residues\n\nTwo-sided permutation test of the difference in means (10,000 label permutations, all residues), Benjamini–Hochberg q-values, Cohen's d."),
    code("""features = cfg["features"] + ["betweenness"]
df_full[features].groupby(df_full["mutation_observed"]).mean().T.rename(columns={0: "mean_no_mutation", 1: "mean_mutation"})"""),
    code("""rng = np.random.default_rng(cfg["seed"])
y = df_full.mutation_observed
rows = []
for f in features:
    x1, x0 = df_full.loc[y == 1, f].dropna().values, df_full.loc[y == 0, f].dropna().values
    rows.append({"feature": f, "mean_mutation": x1.mean(), "mean_other": x0.mean(), "cohens_d": stats.cohens_d(x1, x0),
                 "permutation_p": stats.permutation_mean_diff(x1, x0, 10_000, rng)})
fs = pd.DataFrame(rows)
fs["q_BH"] = stats.bh_fdr(fs.permutation_p.values)
fs["significance"] = fs.q_BH.map(F.stars)
stored_fs = pd.read_csv(table("feature_statistics.csv"))
print("identical to results/tables/feature_statistics.csv:", np.allclose(stored_fs.permutation_p, fs.permutation_p))
print("features with q < 0.05:", int((fs.q_BH < 0.05).sum()), "of", len(fs))
fs.sort_values("permutation_p")"""),
    md("### Supplementary Figure S1"),
    code("""F.figS_feature_boxplots(df_full, stored_fs.set_index("feature"), features)
plt.show()"""),
    md("## 17. Reproducing the earlier 811-residue analysis\n\nApplying the earlier filters (zero betweenness + hard-coded list, log10 betweenness) reproduces the published group sizes exactly, which confirms that only the residue filtering changed:"),
    code("""legacy = df_graph_features[(df_graph_features.betweenness > 0) & ~df_graph_features.index.isin(old_disconnected)].copy()
legacy["log_betweenness"] = np.log10(legacy.betweenness)
print("residues:", len(legacy), "  mutation positions:", int(df_mutation.reindex(legacy.index).fillna(0).sum()))
print("criticality groups:", network.criticality_groups(legacy, 4, cfg["seed"]).replace({"Other": "None"}).value_counts().to_dict())
print("supercritical hubs:", sorted(network.supercritical(legacy, 10, standardise=False).nsmallest(10).index.tolist()))"""),
    md("## Summary"),
    code("""ng = V["network_groups"]
print(f"RIN: {g.vcount()} residues, {g.ecount():,} contacts ({len(typed):,} typed interactions), one connected component")
print(f"mutation positions: {int(y.sum())} of {len(y)}")
for k, v in ng["groups"].items():
    print(f"{k}: {v['n']} residues, {v['n_mutation_positions']} mutation positions ({100 * v['fraction_mutated']:.1f}%)")
print("supercritical hubs:", ", ".join(ng["supercritical_hubs"]))
print(f"hubs at mutation positions: {ng['n_supercritical_mutated']}/10 ->", ", ".join(ng["supercritical_mutation_positions"]))
print(f"features differing between mutation and other positions (q < 0.05): {ng['n_features_significant_q05']} of 15")"""),
]

# =============================================================================================
# 02  Classification models
# =============================================================================================
NB02 = [
    md("""# 02 · Binary classification of mutation positions

Follows the original `classification_model` notebook: load the dataset → explore the features → baseline models → hyperparameter search → final evaluation → ROC curves, confusion matrices and classification reports → feature importance.

**What changed from the original notebook:**
- all 844 residues are used (not 811);
- one 85/15 train/test split is replaced by **repeated nested cross-validation** (5 repeats × 10 outer folds, 5-fold inner grid search on ROC AUC), so every residue is scored by models that never saw it;
- median imputation, standardisation and ADASYN oversampling are fitted inside the training folds only (no PCA);
- predictions for the consensus vote come from these out-of-fold probabilities, not from models refitted on all residues."""),
    SETUP,
    md("## 1. Loading the dataset"),
    code("""df_final = read_processed("residue_features.csv").set_index("residue")
print(df_final.shape)
df_final.head()"""),
    code("""print("Distribution of target values (%):\\n", df_final.mutation_observed.value_counts(normalize=True).mul(100).round(2))
sns.countplot(x="mutation_observed", data=df_final, hue="mutation_observed", legend=False)
plt.title("Distribution of Target Values"); plt.xlabel("Target Value"); plt.ylabel("Count"); plt.show()"""),
    md("Checking for null values (imputed inside the CV folds):"),
    code("df_final.isnull().sum()[lambda s: s > 0]"),
    md("## 2. Difference of feature means between normal and known mutated residues"),
    code("""features = cfg["features"]
df_final[features].groupby(df_final["mutation_observed"]).mean().T.rename(columns={0: "no mutation", 1: "mutation"})"""),
    md("## 3. Distribution of each feature"),
    code("""fig, axes = plt.subplots(5, 3, figsize=(15, 21)); axes = axes.ravel()
for ax, f in zip(axes, features):
    sns.histplot(df_final[f].dropna(), kde=True, ax=ax)
    ax.axvline(df_final[f].mean(), color="red", linestyle="dashed", linewidth=2, label="Mean")
    ax.axvline(df_final[f].median(), color="green", linestyle="dashed", linewidth=2, label="Median")
    ax.set_title(f"{f} Distribution"); ax.legend()
axes[-1].axis("off"); plt.tight_layout(); plt.show()"""),
    md("## 4. Dividing data into features and labels"),
    code("""X = df_final[features]
Y = df_final["mutation_observed"]
print(X.shape, Y.value_counts().to_dict())
X.describe().T"""),
    md("## 5. Baseline models without hyperparameter tuning\n\n**Why one train/test split is not enough.** The original notebook evaluated each model on a single 85/15 split (122 test residues). Repeating that split with 20 different random seeds shows how much a single split moves the AUC:"),
    code("""from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, ParameterGrid
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
aucs = []
for seed in range(20):
    X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.15, stratify=Y, random_state=seed)
    pipe = ml.make_pipeline("lr", cfg["seed"]).fit(X_tr, y_tr)
    aucs.append(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))
print(f"logistic regression, default settings, 20 different 85/15 splits: AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (range {min(aucs):.3f}-{max(aucs):.3f})")"""),
    md("Untuned baseline models evaluated by 10-fold stratified cross-validation (imputation → scaling → ADASYN → model, fitted on the training folds):"),
    code("""cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg["seed"])
plt.figure(figsize=(7, 6))
base = []
for m in V7_MODEL_ORDER:
    p = cross_val_predict(ml.make_pipeline(m, cfg["seed"], probability=True), X, Y, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(Y, p)
    plt.plot(fpr, tpr, color=MODEL_COLOURS[m], label=f"{MODEL_LABELS[m]} (AUC = {auc(fpr, tpr):.3f})")
    base.append({"model": MODEL_LABELS[m], "untuned 10-fold CV AUC": auc(fpr, tpr)})
plt.plot([0, 1], [0, 1], "k--", label="Random Baseline")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("Baseline models (no tuning)"); plt.legend(loc="lower right"); plt.show()
pd.DataFrame(base)"""),
    md("## 6. Hyperparameter search spaces\n\nSame grids as the original `params` dictionary. kNN `algorithm`/`leaf_size` only change the neighbour-search implementation, so they are not searched; the random forest grid (864 combinations) is sampled with 60 random configurations."),
    code("""for m in MODEL_ORDER:
    n = len(ParameterGrid(ml.GRIDS[m]))
    extra = f"  (randomised search: {cfg['ml']['rf_random_search_iter']} sampled)" if m == "rf" else ""
    print(f"{MODEL_NAMES[m]:22s} {n:4d} configurations{extra}\\n    {ml.GRIDS[m]}")"""),
    md("## 7. Repeated nested cross-validation\n\nThe pipeline and the nested-CV loop used by `scripts/03_train_models.py`:"),
    code("print(inspect.getsource(ml.make_pipeline)); print(inspect.getsource(ml.tune)); print(inspect.getsource(ml.nested_cv))"),
    md("The full run (5 models × 5 repeats × 10 outer folds) takes about 45 minutes, so the notebook loads its stored output. Set `RUN_FULL_NESTED_CV = True` to recompute it here."),
    code("""RUN_FULL_NESTED_CV = False
mc = cfg["ml"]
if RUN_FULL_NESTED_CV:
    long, params = ml.nested_cv(X, Y, MODEL_ORDER, mc["outer_folds"], mc["outer_repeats"], mc["inner_folds"], cfg["seed"], mc["rf_random_search_iter"], -1)
else:
    long = read_processed("oof_predictions_long.csv")
    params = pd.read_csv(table("selected_hyperparameters.csv"))
print("out-of-fold predictions:", long.shape, "  selected hyperparameters:", params.shape)
long.head()"""),
    md("**Reproducibility check:** rerun the first repeat for logistic regression and kNN here and compare with the stored probabilities."),
    code("""t0 = time.time()
demo, demo_params = ml.nested_cv(X, Y, ["lr", "knn"], mc["outer_folds"], 1, mc["inner_folds"], cfg["seed"], mc["rf_random_search_iter"], -1, log=lambda *a: None)
ref = long[(long.repeat == 0) & long.model.isin(["lr", "knn"])]
cmp = demo.merge(ref, on=["model", "residue"], suffixes=("_rerun", "_stored"))
print(f"rerun took {time.time() - t0:.0f} s; max |rerun - stored| probability = {(cmp.probability_rerun - cmp.probability_stored).abs().max():.2e}")"""),
    md("Hyperparameters selected in the 50 outer folds:"),
    code("""display(params.groupby("model")[["inner_cv_auc", "outer_fold_auc"]].agg(["mean", "std"]).round(3))
params.groupby("model").params.agg(lambda s: f"{s.value_counts().iloc[0]}/{len(s)} folds: {s.value_counts().index[0]}").to_frame("most frequently selected")"""),
    md("## 8. ROC curve of each model\n\nEach residue's probability is the mean of its 5 out-of-fold predictions (one per repeat)."),
    code("""prob = read_processed("oof_probabilities.csv").set_index("residue")
yv = prob.mutation_observed.values
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
for ax, m in zip(axes, V7_MODEL_ORDER):
    fpr, tpr, _ = roc_curve(yv, prob[m])
    ax.plot(fpr, tpr, color="red", lw=2, label=f"{MODEL_SHORT[m]} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(f"ROC Curve of {MODEL_LABELS[m]}")
    ax.legend(loc="lower right"); ax.grid(alpha=0.4)
plt.tight_layout(); plt.show()"""),
    md("### Figure 4a — all models"),
    code("""F.fig4a_roc(prob)
plt.show()"""),
    md("### Supplementary Figure S4 — mean ± SD ROC across the 50 outer folds"),
    code("""F.figS_cv_roc(long)
plt.show()"""),
    md("## 9. Performance and DeLong comparisons\n\nPooled out-of-fold AUC with DeLong 95% CI, recomputed here and compared with `model_performance.csv`:"),
    code("""rows = []
for m in V7_MODEL_ORDER:
    a, lo, hi = stats.delong_ci(yv, prob[m].values)
    rows.append({"model": MODEL_LABELS[m], "pooled OOF AUC": a, "95% CI low": lo, "95% CI high": hi,
                 "fold AUC mean": params.loc[params.model == m, "outer_fold_auc"].mean(), "fold AUC SD": params.loc[params.model == m, "outer_fold_auc"].std()})
display(pd.DataFrame(rows).sort_values("pooled OOF AUC", ascending=False))
pd.read_csv(table("delong_comparisons.csv"))"""),
    md("## 10. Confusion matrices and classification reports\n\nTwo cut-offs per model: probability ≥ 0.5, and the model's 90%-specificity threshold used for the consensus vote (notebook 03). Because ADASYN balances the training classes, the models are calibrated to 50% prevalence and a 0.5 cut-off flags about a third of all unannotated residues."),
    code("""thresholds = pd.read_csv(table("vote_thresholds.csv")).set_index("model")["threshold"].to_dict()
for m in V7_MODEL_ORDER:
    for name, thr in [("probability >= 0.5", 0.5), ("90% specificity", thresholds[m])]:
        pred = (prob[m] >= thr).astype(int)
        print(f"===== {MODEL_LABELS[m]} | {name} (threshold {thr:.3f})")
        print(pd.DataFrame(confusion_matrix(yv, pred), index=["true: no mutation", "true: mutation"], columns=["pred: no mutation", "pred: mutation"]))
        print(classification_report(yv, pred, target_names=["No mutation", "Mutation"], digits=3))"""),
    md("### Supplementary Figure S3 — confusion matrices"),
    code("""F.figS_confusion(prob, thresholds)
plt.show()"""),
    md("### Supplementary Figure S2 — accuracy, F1, AUC and balanced accuracy per outer fold (mean ± SD, threshold 0.5)"),
    code("""fm = F.fold_metrics(long, cfg["ml"]["vote_threshold"])
display(fm.groupby("model")[["accuracy", "f1", "roc_auc", "balanced_accuracy"]].agg(["mean", "std"]).round(3))
F.figS_model_metrics(fm)
plt.show()"""),
    md("## 11. Feature importance\n\nPermutation importance: drop in held-out ROC AUC when one feature is shuffled in the test fold (first CV repeat, 20 permutations)."),
    code("""pi = pd.read_csv(table("permutation_importance.csv")).set_index("feature")
display(pi.round(4))
F.figS_permutation_importance(pi)
plt.show()"""),
    md("SHAP values of the models refitted on all residues (interpretation only, not used for evaluation):"),
    code("""for m in ["rf", "lr", "xgb"]:
    sv = read_processed(f"shap_values_{m}.csv").set_index("residue")
    xin = read_processed(f"shap_inputs_{m}.csv").set_index("residue")[sv.columns]
    F.figS_shap(sv, xin, f"SHAP values — {MODEL_NAMES[m]}")
    plt.show()"""),
    md("## 12. Feature-group ablation\n\nHow much each feature group contributes (nested CV, 2 repeats):"),
    code("""ab = pd.read_csv(table("feature_ablation.csv"))
display(ab.round(4))
F.figS_ablation_vote_rule(ab, pd.read_csv(table("vote_threshold_sensitivity.csv")))
plt.show()"""),
    md("## 13. Sensitivity analysis: the original 811-residue subset"),
    code("""lp = pd.read_csv(table("legacy811_model_performance.csv")).set_index("model")
perf = pd.read_csv(table("model_performance.csv")).set_index("model")
display(pd.DataFrame({"844 residues": perf.pooled_oof_auc, "811 residues": lp.pooled_oof_auc}).round(3))
F.figS_legacy811(perf, lp)
plt.show()"""),
]

# =============================================================================================
# 03  Hotspots
# =============================================================================================
NB03 = [
    md("""# 03 · Consensus hotspots

Follows the "best models" and "comparing model prediction and clinical mutation" sections of the original notebook, using the out-of-fold probabilities from notebook 02:

1. model-specific vote thresholds (90% specificity on residues without a reported mutation)
2. votes per model and the number of models voting (`pred_all`)
3. consensus (≥ 3 of 5) and novel hotspots
4. comparison with the clinical mutation list (Venn, Fig. S6)
5. domains, active-site proximity, stability across CV repeats, vote-rule sensitivity, overlap with draft v7"""),
    SETUP,
    md("## 1. Vote thresholds\n\nA model votes for a residue when its mean out-of-fold probability is at or above the probability that gives 90% specificity on the 630 residues without a reported mutation, so each model flags at most 10% of them."),
    code("""prob = read_processed("oof_probabilities.csv").set_index("residue")
y = prob.mutation_observed
spec = cfg["ml"]["vote_specificity"]
thresholds = {m: float(np.quantile(prob.loc[y == 0, m], spec)) for m in V7_MODEL_ORDER}
pd.DataFrame({"threshold": thresholds,
              "sensitivity": {m: float((prob.loc[y == 1, m] >= t).mean()) for m, t in thresholds.items()},
              "specificity": {m: float((prob.loc[y == 0, m] < t).mean()) for m, t in thresholds.items()},
              "residues flagged": {m: int((prob[m] >= t).sum()) for m, t in thresholds.items()}}).rename(index=MODEL_LABELS)"""),
    md("## 2. Votes of each model and `pred_all`"),
    code("""votes = pd.DataFrame({f"pred_{MODEL_LABELS[m]}": (prob[m] >= thresholds[m]).astype(int) for m in V7_MODEL_ORDER})
votes["pred_all"] = votes.sum(axis=1)
votes["pred_all"].value_counts().sort_index().to_frame("residues")"""),
    code("""seq = data.structure_sequence()
for k in range(5, 0, -1):
    r = votes.index[votes.pred_all == k]
    print(f"pred_all == {k}  ({len(r)} residues):\\n", " ".join(f"{seq[i]}{i}" for i in r), "\\n")"""),
    code("""stored = pd.read_csv(table("residue_votes.csv")).set_index("residue")
print("identical to results/tables/residue_votes.csv:", bool((stored.votes == votes.pred_all).all()))
stored["class"].value_counts().to_frame("residues")"""),
    md("## 3. Comparing model prediction and clinical mutation"),
    code("""def find_counts(list1, list2):
    set1, set2 = set(list1), set(list2)
    print(f"Count of common numbers: {len(set1 & set2)}")
    print(f"Count of unique numbers in list1 (clinical only): {len(set1 - set2)}")
    print(f"Count of unique numbers in list2 (model only = novel hotspots): {len(set2 - set1)}")
    print(f"Unique numbers in list2: {sorted(set2 - set1)}")

known = y.index[y == 1].tolist()
predicted = votes.index[votes.pred_all >= cfg["ml"]["consensus_min_votes"]].tolist()
find_counts(known, predicted)"""),
    md("### Supplementary Figure S6"),
    code("""F.figS_predicted_vs_clinical(stored, cfg["ml"]["consensus_min_votes"])
plt.show()"""),
    md("## 4. Novel consensus hotspots"),
    code("""novel = pd.read_csv(table("novel_hotspots.csv")).set_index("residue")
print(len(novel), "novel hotspots; votes:", novel.votes.value_counts().sort_index().to_dict())
novel[["residue_label", "domain", "votes", "mean_probability", "consensus_repeat_fraction", "criticality", "areaSAS", "conservation",
       "nearest_active_site_residue", "nearest_active_site_distance_A"]].round(3)"""),
    md("## 5. Domain distribution\n\nObserved number of novel hotspots per domain against the number expected from domain size (two-sided binomial test):"),
    code("pd.read_csv(table(\"hotspot_domain_breakdown.csv\")).round(3)"),
    md("## 6. Active-site proximity\n\nMinimum heavy-atom distance to the catalytic nucleophile D518, the acid/base D616 and the substrate-binding residues:"),
    code("""print("active-site residues among the novel hotspots:", novel.loc[novel.is_active_site_residue, "residue_label"].tolist())
print("novel hotspots within 8 Å of an active-site residue:", int((novel.nearest_active_site_distance_A <= 8).sum()), "of", len(novel))
sns.histplot(novel.nearest_active_site_distance_A, bins=15)
plt.xlabel("distance to nearest active-site residue (Å)"); plt.ylabel("novel hotspots"); plt.show()"""),
    md("## 7. Stability across the five CV repeats\n\nThe consensus vote repeated separately within each CV repeat:"),
    code("""print("novel hotspots reaching consensus in all 5 repeats:", int((novel.consensus_repeat_fraction == 1).sum()), "of", len(novel))
F.figS_hotspot_stability(novel)
plt.show()"""),
    md("## 8. Sensitivity to the vote rule"),
    code("pd.read_csv(table(\"vote_threshold_sensitivity.csv\"))"),
    md("## 9. Overlap with the 42 hotspots of draft v7"),
    code("""V7 = [92, 135, 136, 244, 247, 258, 280, 282, 296, 302, 315, 317, 331, 338, 344, 353, 354, 364, 365, 370, 378, 399, 402, 407, 443,
      488, 516, 518, 597, 606, 639, 673, 683, 703, 710, 720, 722, 747, 754, 821, 833, 951]
shared = sorted(set(V7) & set(novel.index))
print(f"{len(shared)} of 42 v7 hotspots are novel consensus hotspots here")
lost = sorted(set(V7) - set(novel.index))
display(stored.loc[lost, ["wt", "votes", "class", "mean_probability"]])
print("new novel hotspots not in the v7 list:", [f"{seq[r]}{r}" for r in sorted(set(novel.index) - set(V7))])"""),
    md("## 10. Known mutation positions that no model prioritises"),
    code("""known_df = stored[stored.mutation_observed == 1]
print("known positions with 0 votes:", int((known_df.votes == 0).sum()), "  below consensus:", int((known_df.votes < 3).sum()), "of", len(known_df))
pd.read_csv(table("discordant_positions.csv")).round(4)"""),
]

# =============================================================================================
# 04  Substitution Severity Score
# =============================================================================================
NB04 = [
    md("""# 04 · Substitution Severity Score (SSS)

SSS(i → j) = [ EVOscore(i → j) + DI(i, j) ] / 2, with both terms scaled to 0–1.

- **EVOscore** = −log10(f_variant / f_wildtype) from the ConSurf multiple-sequence alignment of GAA
- **DI** = physicochemical distance between the two amino acids, from 544 AAindex properties (SeqinR)

No EVE information enters the SSS (the earlier benchmark used an EVE-based score)."""),
    SETUP,
    md("## 1. ConSurf alignment frequencies\n\nThe ConSurf grades file lists, for every position, the residue variety observed in the alignment:"),
    code("""lines = raw("conservation", "5NN3_consurf_grades.txt").read_text().splitlines()
print(*[l[:175] for l in lines[:20]], sep="\\n")"""),
    code("""freq = data.consurf_frequencies()      # positions x 20 amino-acid frequencies
print(freq.shape)
freq.head()"""),
    code("""seq = read_processed("residue_features.csv").set_index("residue")["wt"]     # the 844 RIN residues
common = freq.index.intersection(seq.index)
print("positions in ConSurf:", len(freq), "  residues in the RIN:", len(seq), "  shared:", len(common))
print("wild-type mismatches between alignment and structure:", int((freq.loc[common, "wt_consurf"] != seq[common]).sum()))
print("\\nexample, catalytic nucleophile D518:", {k: v for k, v in freq.loc[518, AA].items() if v > 0})"""),
    md("## 2. EVOscore, step by step for D518\n\nAmino acids never observed at a position get f = 0.001 (0.1%); `<1%` entries are taken as 0.5%. EVOscore is clipped to [0, 3] (3 = −log10(0.001)) and divided by 3."),
    code("""floor, cap = cfg["sss"]["min_frequency"], cfg["sss"]["evoscore_cap"]
p, w = 518, seq[518]
f = freq.loc[p, AA].astype(float).clip(lower=floor)
evo = -np.log10(f / f[w])
pd.DataFrame({"f (alignment)": freq.loc[p, AA], "f used": f, "EVOscore": evo, "EVOscore scaled": evo.clip(0, cap) / cap}).T.round(3)"""),
    code("""evo_all = sss_mod.evoscore(freq.loc[common], seq[common], floor)
vals = evo_all.values[~np.isclose(evo_all.values, 0)]
print(f"EVOscore over all substitutions: min {vals.min():.2f}, median {np.median(vals):.2f}, max {vals.max():.2f}")
print(f"fraction negative (variant more frequent than wild type): {(vals < 0).mean():.3f};  fraction at the cap: {(vals >= cap).mean():.3f}")"""),
    md("## 3. Physicochemical distance index (DI)\n\n544 AAindex properties exported from SeqinR (`scripts/00_export_aaindex.R`):"),
    code("""ai = pd.read_csv(raw("physicochemical", "aaindex1_seqinr.csv"))
print(ai.shape)
display(ai.head())
props = ai[AA]
print("properties with a missing value:", int(props.isna().any(axis=1).sum()), "-> filled with that property's mean")
props = props.apply(lambda r: r.fillna(r.mean()), axis=1)"""),
    md("Standardise each property across the 20 amino acids, then PCA:"),
    code("""from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
z = StandardScaler().fit_transform(props.T.values)     # 20 amino acids x 544 properties
pca = PCA().fit(z)
cum = np.cumsum(pca.explained_variance_ratio_)
k = int(np.searchsorted(cum, cfg["sss"]["pca_variance"]) + 1)
print(f"components needed for >= 99% variance: {k} ({100 * cum[k - 1]:.2f}%)")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, len(cum) + 1), pca.explained_variance_ratio_, color="#1f77b4", label="per component")
ax.plot(range(1, len(cum) + 1), cum, "o-", color="#d62728", label="cumulative")
ax.axhline(0.99, ls="--", color="grey"); ax.axvline(k, ls=":", color="grey")
ax.set_xlabel("principal component"); ax.set_ylabel("explained variance"); ax.legend(); plt.show()"""),
    code("""from scipy.spatial.distance import pdist, squareform
di = pd.DataFrame(squareform(pdist(pca.transform(z)[:, :k])), index=AA, columns=AA)
di_lib, info = sss_mod.distance_index(cfg["sss"]["pca_variance"])
print("identical to the pipeline:", np.allclose(di, di_lib)); print(info)
legacy = pd.read_csv(raw("physicochemical", "legacy_DI_matrix.csv")).dropna(subset=["Reference_aa"])
L = legacy.pivot(index="Reference_aa", columns="Variety_aa", values="Physiochem_score").loc[AA, AA]
iu = np.triu_indices(20, 1)
print(f"Pearson r with the DI matrix used in earlier drafts: {np.corrcoef(di.values[iu], L.values[iu])[0, 1]:.4f}")"""),
    md("### Supplementary Figure S8 — DI matrix and EVOscore distribution"),
    code("""sss = sss_mod.severity_long(freq, seq[common], di, floor, cap)
title = f"Distance index: {info['n_properties']} AAindex properties, {k} PCs ({100 * cum[k - 1]:.1f}% variance)"
F.figS_di_evoscore(di, sss, cap, title)
plt.show()"""),
    md("## 4. SSS for every possible substitution"),
    code("""stored = read_processed("sss_all_substitutions.csv")
print(len(sss), "substitutions at", sss.residue.nunique(), "residues; identical to stored file:", np.allclose(sss.sss.values, stored.sss.values))
assert np.allclose(sss.sss, (sss.evoscore_scaled + sss.di_scaled) / 2)
sss.head()"""),
    code("""cls = np.select([sss.sss < 0.5, sss.sss <= 0.6], ["Benign (< 0.5)", "Unknown (0.5-0.6)"], "Severe (> 0.6)")
print(pd.Series(cls).value_counts().to_string())
print("\\nmost severe substitutions:")
sss.sort_values("sss", ascending=False).head(15)[["variant", "f_wt", "f_mt", "evoscore", "di", "sss"]]"""),
    md("### Figure 5a — SSS framework (DI matrix shown, not BLOSUM62)"),
    code("""F.fig5a_schematic(di, info["n_properties"], k, cum[k - 1])
plt.show()"""),
    md("### Figure 5b — SSS at the novel consensus hotspots\n\nGrey cells = wild-type residue."),
    code("""novel = pd.read_csv(table("novel_hotspots.csv")).set_index("residue")
feat = read_processed("residue_features.csv").set_index("residue")
F.fig5b_novel_heatmap(sss, novel, feat)
plt.show()
mean_novel = sss[sss.residue.isin(novel.index)].groupby("residue").sss.mean()
print(f"mean SSS at novel hotspots {mean_novel.mean():.3f} vs all other residues {sss[~sss.residue.isin(novel.index)].groupby('residue').sss.mean().mean():.3f}")"""),
    md("### Supplementary Figure S9 — SSS across the whole protein"),
    code("""F.figS_full_protein_sss(sss)
plt.show()"""),
]

# =============================================================================================
# 05  Benchmark
# =============================================================================================
NB05 = [
    md("""# 05 · Benchmark of SSS against AlphaMissense, ESM-1v and EVE

ClinVar GAA missense variants (the snapshot distributed with the EVE GAA model). P/LP = pathogenic or likely pathogenic, B/LB = benign or likely benign. The benchmark set contains the P/LP and B/LB variants scored by all four predictors."""),
    SETUP,
    md("## 1. ClinVar labels"),
    code("""eve = pd.read_csv(raw("predictors", "EVE_LYAG_HUMAN.csv"), usecols=["wt_aa", "position", "mt_aa", "ClinVar_ClinicalSignificance", "EVE_scores_ASM"])
print("EVE file:", eve.shape)
clin = eve.dropna(subset=["ClinVar_ClinicalSignificance"]).rename(columns={"wt_aa": "wt", "position": "residue", "mt_aa": "mt",
                                                                          "ClinVar_ClinicalSignificance": "clinvar", "EVE_scores_ASM": "EVE"})
clin.clinvar.value_counts().to_frame("variants")"""),
    code("""b = cfg["benchmark"]
def klass(s):
    if s in b["pathogenic"]: return "Pathogenic"
    if s in b["benign"]: return "Benign"
    if s.startswith("Uncertain"): return "VUS"
    if s.startswith("Conflicting"): return "Conflicting"
    return "Other"
clin["class"] = clin.clinvar.map(klass)
clin["class"].value_counts().to_frame("variants")"""),
    md("## 2. Predictor scores\n\nESM-1v reports a log-likelihood ratio (lower = more damaging), so it is negated; for every score, higher = more damaging."),
    code("""am = pd.read_csv(raw("predictors", "AlphaMissense_P10253.tsv"), sep="\\t").rename(columns={"position": "residue", "a.a.1": "wt", "a.a.2": "mt", "pathogenicity score": "AlphaMissense"})
esm = pd.read_csv(raw("predictors", "ESM1v_P10253.csv"))
esm["wt"], esm["mt"], esm["residue"], esm["ESM-1v"] = esm.variant.str[0], esm.variant.str[-1], esm.pos, -esm.score
sss = read_processed("sss_all_substitutions.csv").rename(columns={"sss": "SSS", "evoscore_scaled": "EVOscore", "di_scaled": "DI"})
v = (clin.merge(am[["residue", "wt", "mt", "AlphaMissense"]], on=["residue", "wt", "mt"], how="left")
         .merge(esm[["residue", "wt", "mt", "ESM-1v"]], on=["residue", "wt", "mt"], how="left")
         .merge(sss[["residue", "wt", "mt", "SSS", "EVOscore", "DI"]], on=["residue", "wt", "mt"], how="left"))
tools = ["SSS", "AlphaMissense", "ESM-1v", "EVE"]
v["all_four_scores"] = v[tools].notna().all(axis=1)
binary = v[v["class"].isin(["Pathogenic", "Benign"])]
print("P/LP + B/LB variants:", len(binary))
print("coverage:", {t: int(binary[t].notna().sum()) for t in tools}, "(SSS needs a residue in the structure, 81-952)")
bench = binary[binary.all_four_scores].copy()
bench["y"] = (bench["class"] == "Pathogenic").astype(int)
print(f"benchmark set: {len(bench)} variants = {bench.y.sum()} P/LP + {(1 - bench.y).sum()} B/LB")"""),
    md("## 3. Discrimination\n\nROC AUC with DeLong and stratified-bootstrap (2,000 resamples) 95% CIs, Spearman correlation with the class, Mann–Whitney U test. EVOscore and DI are the two SSS components."),
    code("""from scipy import stats as ss
rng = np.random.default_rng(cfg["seed"])
yb = bench.y.values
rows = []
for t in tools + ["EVOscore", "DI"]:
    s = bench[t].values
    a, lo, hi = stats.delong_ci(yb, s)
    blo, bhi = stats.bootstrap_auc_ci(yb, s, b["bootstrap"], rng)
    rho, rp = ss.spearmanr(s, yb)
    rows.append({"predictor": t, "ROC AUC": a, "DeLong CI": f"{lo:.3f}-{hi:.3f}", "bootstrap CI": f"{blo:.3f}-{bhi:.3f}",
                 "Spearman r": rho, "Spearman P": rp, "Mann-Whitney P": ss.mannwhitneyu(s[yb == 1], s[yb == 0]).pvalue})
perf = pd.DataFrame(rows)
stored = pd.read_csv(table("benchmark_performance.csv"))
print("identical to results/tables/benchmark_performance.csv:", np.allclose(perf["ROC AUC"], stored.roc_auc))
perf"""),
    md("Paired DeLong tests of SSS against each predictor:"),
    code("""pd.DataFrame([{"comparison": f"SSS vs {t}", **stats.delong_test(yb, bench.SSS.values, bench[t].values)} for t in ["AlphaMissense", "ESM-1v", "EVE", "EVOscore", "DI"]]).round(4)"""),
    md("### Figure 6 — ROC curves and summary table"),
    code("""bperf = stored.set_index("predictor")
F.fig6_benchmark(bench, bperf, pd.read_csv(table("benchmark_delong_vs_sss.csv")))
plt.show()"""),
    md("### Supplementary Figure S11 — AUC ranking and score separation"),
    code("""F.figS_benchmark_separation(bench, bperf)
plt.show()"""),
    md("## 4. SSS classification thresholds\n\nThe 0.5 and 0.6 cut-offs of the severity classes (Fig. 5a) and the Youden-optimal cut-off (derived from the same variants, so optimistic):"),
    code("""from sklearn.metrics import roc_curve
fpr, tpr, thr = roc_curve(yb, bench.SSS)
rows = []
for name, cut in [("0.5", 0.5), ("0.6", 0.6), ("Youden (in-sample)", float(thr[np.argmax(tpr - fpr)]))]:
    pred = (bench.SSS.values >= cut).astype(int)
    tp, fn = int(((pred == 1) & (yb == 1)).sum()), int(((pred == 0) & (yb == 1)).sum())
    tn, fp = int(((pred == 0) & (yb == 0)).sum()), int(((pred == 1) & (yb == 0)).sum())
    rows.append({"cut-off": name, "value": round(cut, 3), "TP": tp, "FN": fn, "TN": tn, "FP": fp,
                 "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "accuracy": (tp + tn) / len(yb)})
pd.DataFrame(rows).round(3)"""),
    md("Benign variants with SSS ≥ 0.5 (false positives at the 0.5 cut-off):"),
    code("bench[(bench.y == 0) & (bench.SSS >= 0.5)][[\"wt\", \"residue\", \"mt\", \"clinvar\", \"SSS\", \"EVOscore\", \"DI\", \"AlphaMissense\", \"EVE\"]].round(3)"),
    md("### Figure 5c and Supplementary Figure S10 — SSS at the ClinVar positions\n\nBlack squares mark the ClinVar substitution itself."),
    code("""feat = read_processed("residue_features.csv").set_index("residue")
sss_long = read_processed("sss_all_substitutions.csv")
F.fig5c_clinvar_heatmap(sss_long, bench, feat)
plt.show()"""),
    code("""F.figS_clinvar_components(sss_long, bench, feat)
plt.show()"""),
    md("## 5. Variants of uncertain significance"),
    code("""vus = v[(v["class"] == "VUS") & v.SSS.notna()].copy()
vus["SSS class"] = np.select([vus.SSS < 0.5, vus.SSS <= 0.6], ["Benign (< 0.5)", "Unknown (0.5-0.6)"], "Severe (> 0.6)")
print(len(vus), "VUS inside the structure"); print(vus["SSS class"].value_counts().to_string())
print("\\nSpearman r between SSS and AlphaMissense on the VUS:", round(ss.spearmanr(vus.SSS, vus.AlphaMissense, nan_policy="omit")[0], 3))
vus.sort_values("SSS", ascending=False).head(20)[["wt", "residue", "mt", "SSS", "AlphaMissense", "ESM-1v", "EVE"]].round(3)"""),
]

NOTEBOOKS = {
    "01_RIN_criticality_analysis.ipynb": NB01,
    "02_classification_models.ipynb": NB02,
    "03_consensus_hotspots.ipynb": NB03,
    "04_substitution_severity_score.ipynb": NB04,
    "05_benchmark_clinvar.ipynb": NB05,
}

if __name__ == "__main__":
    only = set(__import__("sys").argv[1:])
    for old in NB.glob("*.ipynb"):
        if not only or old.name in only:
            old.unlink()
    for name, cells in NOTEBOOKS.items():
        if only and name not in only:
            continue
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.cells = [nbf.v4.new_markdown_cell(c) if kind == "md" else nbf.v4.new_code_cell(c) for kind, c in cells]
        ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB)}})
        nbf.write(nb, NB / name)
        print("executed", name, f"({len(cells)} cells)")
