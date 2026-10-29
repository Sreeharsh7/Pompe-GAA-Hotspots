"""Stage 2: topological criticality groups, supercritical hubs and univariate feature statistics.

Outputs
  data/processed/residue_network_groups.csv
  results/tables/feature_statistics.csv
  results/tables/centrality_correlation.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats as ss

from gaa_hotspot import network, stats
from gaa_hotspot.common import config, get_logger, processed, read_processed, save_csv, table, update_values

log = get_logger("02_network")
cfg = config()
df = read_processed("residue_features.csv").set_index("residue")
y = df["mutation_observed"]

# Criticality groups and supercritical hubs ------------------------------------------------
df["criticality"] = network.criticality_groups(df, cfg["criticality"]["kbins"], cfg["seed"])
n_sc = cfg["criticality"]["n_supercritical"]
df["distance_to_ideal"] = network.supercritical(df, n_sc, standardise=False)
df["distance_to_ideal_z"] = network.supercritical(df, n_sc, standardise=True)
df["supercritical"] = df.index.isin(df["distance_to_ideal"].nsmallest(n_sc).index)
df["supercritical_z"] = df.index.isin(df["distance_to_ideal_z"].nsmallest(n_sc).index)
save_csv(df[["wt", "criticality", "supercritical", "distance_to_ideal", "supercritical_z", "distance_to_ideal_z", "mutation_observed"]].reset_index(),
         processed("residue_network_groups.csv"))

groups = {}
for g in ["HDHB", "LDHB", "LDLB"]:
    sel = df.criticality == g
    groups[g] = {"n": int(sel.sum()), "n_mutation_positions": int(y[sel].sum()), "fraction_mutated": float(y[sel].mean())}
sc = df[df.supercritical]
hubs = [f"{w}{r}" for r, w in sc["wt"].items()]
sc_mut = sc[sc.mutation_observed == 1]
hdhb_sc = int((sc.criticality == "HDHB").sum())
log.info("groups %s", {k: v["n"] for k, v in groups.items()})
log.info("supercritical hubs %s; %d/%d are mutation positions", hubs, len(sc_mut), n_sc)

# Group-wise SASA and conservation (Fig. 4c,d) ---------------------------------------------
grp_desc = {}
for g in ["HDHB", "LDHB", "LDLB"]:
    sub = df[df.criticality == g]
    grp_desc[g] = {"areaSAS_median": float(sub.areaSAS.median()), "conservation_median": float(sub.conservation.median())}
grp_desc["SC"] = {"areaSAS_median": float(sc.areaSAS.median()), "conservation_median": float(sc.conservation.median())}
kw_sas = ss.kruskal(*[df.loc[df.criticality == g, "areaSAS"] for g in ["HDHB", "LDHB", "LDLB"]])
kw_con = ss.kruskal(*[df.loc[df.criticality == g, "conservation"] for g in ["HDHB", "LDHB", "LDLB"]])

# Univariate comparison: mutation positions vs other residues ------------------------------
rng = np.random.default_rng(cfg["seed"])
features = cfg["features"] + ["betweenness"]
rows = []
for f in features:
    x = df.loc[y == 1, f].dropna().values
    z = df.loc[y == 0, f].dropna().values
    rows.append({
        "feature": f, "mean_mutation_positions": x.mean(), "mean_other": z.mean(),
        "median_mutation_positions": np.median(x), "median_other": np.median(z),
        "cohens_d": stats.cohens_d(x, z),
        "permutation_p": stats.permutation_mean_diff(x, z, 10_000, rng),
        "mann_whitney_p": ss.mannwhitneyu(x, z, alternative="two-sided").pvalue,
        "n_mutation_positions": len(x), "n_other": len(z),
    })
fs = pd.DataFrame(rows)
fs["permutation_q_bh"] = stats.bh_fdr(fs.permutation_p.values)
save_csv(fs, table("feature_statistics.csv"))

cent_cols = ["betweenness", "page_rank", "degree", "burt", "k_core", "closeness", "authority"]
corr = df[cent_cols].corr(method="spearman")
save_csv(corr.reset_index().rename(columns={"index": "centrality"}), table("centrality_correlation.csv"))

update_values(
    "network_groups",
    {
        "groups": groups,
        "supercritical_hubs": hubs,
        "supercritical_mutation_positions": [f"{w}{r}" for r, w in sc_mut["wt"].items()],
        "n_supercritical_mutated": int(len(sc_mut)),
        "n_supercritical_in_HDHB": hdhb_sc,
        "supercritical_hubs_zscored": [f"{w}{r}" for r, w in df[df.supercritical_z]["wt"].items()],
        "n_hubs_shared_raw_vs_z": int((df.supercritical & df.supercritical_z).sum()),
        "fraction_mutation_positions_that_are_hubs": float(len(sc_mut) / y.sum()),
        "group_medians": grp_desc,
        "kruskal_areaSAS_p": float(kw_sas.pvalue),
        "kruskal_conservation_p": float(kw_con.pvalue),
        "n_features_significant_q05": int((fs.permutation_q_bh < 0.05).sum()),
        "centrality_spearman_range": [float(corr.where(~np.eye(len(corr), dtype=bool)).min().min()), float(corr.where(~np.eye(len(corr), dtype=bool)).max().max())],
    },
)
print(fs[["feature", "mean_mutation_positions", "mean_other", "cohens_d", "permutation_p", "permutation_q_bh"]].round(4).to_string(index=False))
