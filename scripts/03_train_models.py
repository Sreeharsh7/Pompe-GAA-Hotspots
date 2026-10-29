"""Stage 3: repeated nested cross-validation of the five classifiers.

Outputs (prefix 'legacy811_' when --legacy811 is given)
  data/processed/oof_predictions_long.csv   every out-of-fold probability (repeat x residue x model)
  data/processed/oof_probabilities.csv      mean OOF probability per residue and model
  results/tables/selected_hyperparameters.csv
  results/tables/model_performance.csv
  results/tables/delong_comparisons.csv

Usage
  python scripts/03_train_models.py                 # main analysis, all 844 residues
  python scripts/03_train_models.py --legacy811     # sensitivity: original 811-residue subset
  python scripts/03_train_models.py --repeats 1 --folds 1   # quick smoke test
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from gaa_hotspot import ml, stats
from gaa_hotspot.common import MODEL_NAMES, MODEL_ORDER, config, get_logger, processed, read_processed, save_csv, table, update_values

# Residues removed in the original analysis: zero betweenness plus a hard-coded
# 'disconnected' list (8 of which are in fact connected in the RIN).
LEGACY_EXTRA = [97, 206, 337, 288, 88, 831, 483, 651, 758, 814, 678, 799, 848, 946, 914, 924]

ap = argparse.ArgumentParser()
ap.add_argument("--legacy811", action="store_true")
ap.add_argument("--repeats", type=int)
ap.add_argument("--folds", type=int, help="stop after this many outer folds (smoke test)")
ap.add_argument("--jobs", type=int, default=-1)
args = ap.parse_args()

cfg = config()
mcfg = cfg["ml"]
log = get_logger("03_models")
prefix = "legacy811_" if args.legacy811 else ""

df = read_processed("residue_features.csv").set_index("residue")
if args.legacy811:
    df = df[(df.betweenness > 0) & ~df.index.isin(LEGACY_EXTRA)].copy()
    df["log_betweenness"] = np.log10(df["betweenness"])
X = df[cfg["features"]]
y = df["mutation_observed"]
log.info("%s%d residues, %d positives, %d features", prefix, len(X), y.sum(), X.shape[1])

repeats = args.repeats or mcfg["outer_repeats"]
t0 = time.time()
if args.folds:
    # smoke test: time a limited number of outer folds of the first repeat
    from sklearn.model_selection import StratifiedKFold
    outer = StratifiedKFold(n_splits=mcfg["outer_folds"], shuffle=True, random_state=cfg["seed"])
    for f, (tr, te) in enumerate(outer.split(X, y)):
        if f >= args.folds:
            break
        for m in MODEL_ORDER:
            t = time.time()
            best, inner = ml.tune(m, X.values[tr], y.values[tr], cfg["seed"], mcfg["inner_folds"], mcfg["rf_random_search_iter"], args.jobs)
            p = ml.fit_predict(m, best, X.values[tr], y.values[tr], X.values[te], cfg["seed"])
            log.info("%s inner AUC %.3f  outer fold AUC %.3f  %.1fs  %s", m, inner, roc_auc_score(y.values[te], p), time.time() - t, best)
    sys.exit(0)

oof, params = ml.nested_cv(X, y, MODEL_ORDER, mcfg["outer_folds"], repeats, mcfg["inner_folds"], cfg["seed"], mcfg["rf_random_search_iter"], args.jobs, log=log.info)
log.info("nested CV finished in %.1f min", (time.time() - t0) / 60)
save_csv(oof, processed(f"{prefix}oof_predictions_long.csv"))
save_csv(params, table(f"{prefix}selected_hyperparameters.csv"))

# Mean OOF probability per residue (every value is out-of-fold for that residue)
wide = oof.groupby(["residue", "model"])["probability"].mean().unstack()[MODEL_ORDER]
wide.insert(0, "mutation_observed", y.reindex(wide.index).values)
save_csv(wide.reset_index(), processed(f"{prefix}oof_probabilities.csv"))

# Performance ------------------------------------------------------------------------------
yv = wide["mutation_observed"].values
thr = mcfg["vote_threshold"]
perf = []
for m in MODEL_ORDER:
    auc, lo, hi = stats.delong_ci(yv, wide[m].values)
    per_rep = oof[oof.model == m].groupby("repeat").apply(lambda d: roc_auc_score(d.y, d.probability), include_groups=False)
    folds = params[params.model == m]
    row = {"model": m, "model_name": MODEL_NAMES[m], "pooled_oof_auc": auc, "auc_ci_low": lo, "auc_ci_high": hi,
           "per_repeat_auc_mean": per_rep.mean(), "per_repeat_auc_sd": per_rep.std(ddof=1),
           "fold_auc_mean": folds.outer_fold_auc.mean(), "fold_auc_sd": folds.outer_fold_auc.std(ddof=1)}
    row.update(ml.threshold_metrics(yv, wide[m].values, thr))
    perf.append(row)
perf = pd.DataFrame(perf)
save_csv(perf, table(f"{prefix}model_performance.csv"))

best = perf.sort_values("pooled_oof_auc", ascending=False).model.iloc[0]
rows = []
for m in MODEL_ORDER:
    if m == best:
        continue
    r = stats.delong_test(yv, wide[best].values, wide[m].values)
    rows.append({"comparison": f"{MODEL_NAMES[best]} vs {MODEL_NAMES[m]}", "model_a": best, "model_b": m, **r})
dl = pd.DataFrame(rows)
dl["p_holm"] = np.minimum(1, dl.p.values * (len(dl) - np.argsort(np.argsort(dl.p.values))))
save_csv(dl, table(f"{prefix}delong_comparisons.csv"))

update_values(f"{prefix}models", {
    "n_residues": int(len(X)), "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
    "repeats": repeats, "outer_folds": mcfg["outer_folds"], "inner_folds": mcfg["inner_folds"],
    "performance": perf.set_index("model").to_dict(orient="index"),
    "best_model": best,
    "delong": dl.set_index("model_b")[["difference", "z", "p", "p_holm"]].to_dict(orient="index"),
    "runtime_min": (time.time() - t0) / 60,
})
print(perf[["model", "pooled_oof_auc", "auc_ci_low", "auc_ci_high", "per_repeat_auc_mean", "per_repeat_auc_sd", "sensitivity", "specificity", "mcc"]].round(4).to_string(index=False))
print(dl.round(5).to_string(index=False))
