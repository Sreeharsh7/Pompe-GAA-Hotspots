"""Stage 5b: feature-group ablation — does the RIN add information beyond conservation and structure?

Same nested CV as stage 3 (2 repeats x 10 outer folds) for logistic regression and XGBoost
on feature subsets. Output: results/tables/feature_ablation.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.metrics import roc_auc_score

from gaa_hotspot import ml, stats
from gaa_hotspot.common import config, get_logger, read_processed, save_csv, table, update_values

log = get_logger("05b_ablation")
cfg = config()
mcfg = cfg["ml"]
df = read_processed("residue_features.csv").set_index("residue")
y = df["mutation_observed"]

NET = ["degree", "log_betweenness", "closeness", "burt", "authority", "page_rank", "k_core"]
STRUCT = ["kdhydro", "areaSAS", "areaSES", "relSESA", "phi", "psi"]
SUBSETS = {
    "Conservation only": ["conservation"],
    "Network only": NET,
    "Structure only": STRUCT,
    "Structure + conservation": STRUCT + ["conservation"],
    "Network + conservation": NET + ["conservation"],
    "All 14 features": NET + STRUCT + ["conservation"],
}
REPEATS = 2

rows, oof_store = [], {}
for name, cols in SUBSETS.items():
    oof, _ = ml.nested_cv(df[cols], y, ["lr", "xgb"], mcfg["outer_folds"], REPEATS, mcfg["inner_folds"], cfg["seed"], mcfg["rf_random_search_iter"], -1, log=lambda *_: None)
    for m in ["lr", "xgb"]:
        mean = oof[oof.model == m].groupby("residue").probability.mean().reindex(df.index)
        oof_store[(name, m)] = mean
        auc, lo, hi = stats.delong_ci(y.values, mean.values)
        rows.append({"feature_set": name, "n_features": len(cols), "model": m, "pooled_oof_auc": auc, "auc_ci_low": lo, "auc_ci_high": hi})
    log.info("%s done", name)

res = pd.DataFrame(rows)
for m in ["lr", "xgb"]:
    full = oof_store[("All 14 features", m)].values
    for name in SUBSETS:
        if name == "All 14 features":
            continue
        t = stats.delong_test(y.values, full, oof_store[(name, m)].values)
        res.loc[(res.feature_set == name) & (res.model == m), "delong_p_vs_all"] = t["p"]
save_csv(res, table("feature_ablation.csv"))
update_values("ablation", {f"{r.feature_set} | {r.model}": {"auc": r.pooled_oof_auc, "lo": r.auc_ci_low, "hi": r.auc_ci_high, "p_vs_all": getattr(r, "delong_p_vs_all", None)} for r in res.itertuples()})
print(res.round(4).to_string(index=False))
