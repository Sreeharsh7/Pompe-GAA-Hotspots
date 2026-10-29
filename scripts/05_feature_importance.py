"""Stage 5: feature attribution for the trained classifiers.

* Permutation importance (drop in ROC AUC) on held-out outer folds of the first CV repeat,
  for all five models, using the hyperparameters selected inside each fold.
* SHAP values for models refitted on all residues (tree explainer for RF/XGB, linear
  explainer for LR); these describe the fitted models and are not used for evaluation.

Outputs
  results/tables/permutation_importance.csv
  results/tables/shap_importance.csv
  data/processed/shap_values_<model>.csv
  data/processed/final_model_parameters.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from gaa_hotspot import ml
from gaa_hotspot.common import MODEL_ORDER, config, get_logger, processed, read_processed, save_csv, table, update_values

log = get_logger("05_importance")
cfg = config()
mcfg = cfg["ml"]
seed = cfg["seed"]

df = read_processed("residue_features.csv").set_index("residue")
X, y = df[cfg["features"]], df["mutation_observed"].astype(int)
params = pd.read_csv(table("selected_hyperparameters.csv"))

# Permutation importance on held-out folds (repeat 0) -------------------------------------
outer = StratifiedKFold(n_splits=mcfg["outer_folds"], shuffle=True, random_state=seed)
rows = []
for f, (tr, te) in enumerate(outer.split(X, y)):
    for m in MODEL_ORDER:
        p = json.loads(params.query("repeat == 0 and fold == @f and model == @m").params.iloc[0])
        pipe = ml.make_pipeline(m, seed, p, probability=True).fit(X.values[tr], y.values[tr])
        pi = permutation_importance(pipe, X.values[te], y.values[te], scoring="roc_auc", n_repeats=20, random_state=seed, n_jobs=-1)
        for i, feat in enumerate(X.columns):
            rows.append({"model": m, "fold": f, "feature": feat, "auc_drop": pi.importances_mean[i]})
    log.info("permutation importance fold %d done", f + 1)
pi = pd.DataFrame(rows)
agg = pi.groupby(["model", "feature"]).auc_drop.agg(["mean", "std"]).reset_index()
wide = agg.pivot(index="feature", columns="model", values="mean")[MODEL_ORDER]
wide["mean_across_models"] = wide.mean(axis=1)
wide["rank"] = wide.mean_across_models.rank(ascending=False).astype(int)
wide = wide.sort_values("rank")
save_csv(wide.reset_index(), table("permutation_importance.csv"))

# SHAP on final models ---------------------------------------------------------------------
final = ml.final_models(X, y, ["lr", "rf", "xgb"], seed, mcfg["inner_folds"], mcfg["rf_random_search_iter"], -1)
(processed("final_model_parameters.json")).write_text(json.dumps({m: {"params": v["params"], "inner_cv_auc": v["cv_auc"]} for m, v in final.items()}, indent=2, default=str))
shap_rows = []
for m, v in final.items():
    pipe = v["pipeline"]
    Xt = pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(X.values))
    model = pipe.named_steps["model"]
    if m == "lr":
        sv = shap.LinearExplainer(model, Xt).shap_values(Xt)
    else:
        sv = shap.TreeExplainer(model).shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
    sdf = pd.DataFrame(sv, columns=X.columns, index=X.index)
    save_csv(sdf.reset_index(), processed(f"shap_values_{m}.csv"))
    save_csv(pd.DataFrame(Xt, columns=X.columns, index=X.index).reset_index(), processed(f"shap_inputs_{m}.csv"))
    imp = sdf.abs().mean()
    for feat, val in imp.items():
        shap_rows.append({"model": m, "feature": feat, "mean_abs_shap": val})
sh = pd.DataFrame(shap_rows).pivot(index="feature", columns="model", values="mean_abs_shap")[["lr", "rf", "xgb"]]
sh_norm = sh / sh.max()
sh["mean_normalised"] = sh_norm.mean(axis=1)
sh["rank"] = sh.mean_normalised.rank(ascending=False).astype(int)
sh = sh.sort_values("rank")
save_csv(sh.reset_index(), table("shap_importance.csv"))

update_values("importance", {
    "permutation_top5": wide.index[:5].tolist(),
    "shap_top5": sh.index[:5].tolist(),
    "permutation_auc_drop": wide.mean_across_models.round(4).to_dict(),
    "final_model_params": {m: v["params"] for m, v in final.items()},
})
print(wide.round(4).to_string())
print(sh.round(4).to_string())
