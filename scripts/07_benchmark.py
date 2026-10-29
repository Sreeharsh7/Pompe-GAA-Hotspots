"""Stage 7: benchmark SSS against AlphaMissense, ESM-1v and EVE on ClinVar GAA missense variants.

The ClinVar labels are the snapshot distributed with the EVE GAA (LYAG_HUMAN) file.
SSS is computed only from the ConSurf alignment and AAindex properties, so it shares no
input with EVE.

Outputs
  results/tables/benchmark_variants.csv        every ClinVar-annotated missense variant with all scores
  results/tables/benchmark_performance.csv     AUC (DeLong and bootstrap CIs), Spearman, Mann-Whitney
  results/tables/benchmark_delong_vs_sss.csv   paired DeLong tests of SSS against each predictor
  results/tables/benchmark_threshold.csv       SSS classification at the 0.5 cut-off and the Youden optimum
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats as ss
from sklearn.metrics import roc_curve

from gaa_hotspot import stats
from gaa_hotspot.common import config, get_logger, raw, read_processed, save_csv, table, update_values

log = get_logger("07_benchmark")
cfg = config()
bcfg = cfg["benchmark"]
rng = np.random.default_rng(cfg["seed"])

eve = pd.read_csv(raw("predictors", "EVE_LYAG_HUMAN.csv"), usecols=["wt_aa", "position", "mt_aa", "ClinVar_ClinicalSignificance", "EVE_scores_ASM"])
clin = eve.dropna(subset=["ClinVar_ClinicalSignificance"]).rename(columns={"wt_aa": "wt", "position": "residue", "mt_aa": "mt", "ClinVar_ClinicalSignificance": "clinvar", "EVE_scores_ASM": "EVE"})


def klass(s: str) -> str:
    if s in bcfg["pathogenic"]:
        return "Pathogenic"
    if s in bcfg["benign"]:
        return "Benign"
    if s.startswith("Uncertain"):
        return "VUS"
    if s.startswith("Conflicting"):
        return "Conflicting"
    return "Other"


clin["class"] = clin.clinvar.map(klass)

am = pd.read_csv(raw("predictors", "AlphaMissense_P10253.tsv"), sep="\t")
am = am.rename(columns={"position": "residue", "a.a.1": "wt", "a.a.2": "mt", "pathogenicity score": "AlphaMissense"})[["residue", "wt", "mt", "AlphaMissense"]]
esm = pd.read_csv(raw("predictors", "ESM1v_P10253.csv"))
esm["wt"], esm["mt"], esm["residue"] = esm.variant.str[0], esm.variant.str[-1], esm.pos
esm["ESM-1v"] = -esm.score  # ESM-1v log-likelihood ratio: lower = more damaging, so negate
sss = read_processed("sss_all_substitutions.csv").rename(columns={"sss": "SSS", "evoscore_scaled": "EVOscore", "di_scaled": "DI"})

v = clin.merge(am, on=["residue", "wt", "mt"], how="left").merge(esm[["residue", "wt", "mt", "ESM-1v"]], on=["residue", "wt", "mt"], how="left")
v = v.merge(sss[["residue", "wt", "mt", "SSS", "EVOscore", "DI"]], on=["residue", "wt", "mt"], how="left")
v["in_structure"] = v.residue.between(cfg["structure"]["first_residue"], cfg["structure"]["last_residue"]) & v.SSS.notna()
tools = ["SSS", "AlphaMissense", "ESM-1v", "EVE"]
v["all_four_scores"] = v[tools].notna().all(axis=1)
v = v.sort_values(["residue", "mt"]).reset_index(drop=True)
save_csv(v, table("benchmark_variants.csv"))

counts_all = v["class"].value_counts().to_dict()
binary = v[v["class"].isin(["Pathogenic", "Benign"])]
bench = binary[binary.all_four_scores].copy()
bench["y"] = (bench["class"] == "Pathogenic").astype(int)
yb = bench.y.values
log.info("ClinVar variants %d %s; binary %d; benchmark (all four scores) %d = %d P + %d B",
         len(v), counts_all, len(binary), len(bench), yb.sum(), (1 - yb).sum())

coverage = {t: int(binary[t].notna().sum()) for t in tools}
missing_reason = {t: int((binary[t].isna()).sum()) for t in tools}

rows = []
for t in tools + ["EVOscore", "DI"]:
    s = bench[t].values
    auc, lo, hi = stats.delong_ci(yb, s)
    blo, bhi = stats.bootstrap_auc_ci(yb, s, bcfg["bootstrap"], rng)
    rho, rp = ss.spearmanr(s, yb)
    mw = ss.mannwhitneyu(s[yb == 1], s[yb == 0], alternative="two-sided")
    rows.append({"predictor": t, "role": "component of SSS" if t in ("EVOscore", "DI") else "predictor",
                 "n": len(s), "n_pathogenic": int(yb.sum()), "n_benign": int((1 - yb).sum()),
                 "roc_auc": auc, "delong_ci_low": lo, "delong_ci_high": hi, "bootstrap_ci_low": blo, "bootstrap_ci_high": bhi,
                 "spearman_r": rho, "spearman_p": rp, "mann_whitney_p": mw.pvalue})
perf = pd.DataFrame(rows)
save_csv(perf, table("benchmark_performance.csv"))

dl = []
for t in ["AlphaMissense", "ESM-1v", "EVE", "EVOscore", "DI"]:
    r = stats.delong_test(yb, bench["SSS"].values, bench[t].values)
    dl.append({"comparison": f"SSS vs {t}", **r})
dl = pd.DataFrame(dl)
save_csv(dl, table("benchmark_delong_vs_sss.csv"))

# Threshold behaviour of SSS (0.5 = cut-off drawn in Fig. 5a; Youden optimum is descriptive only)
fpr, tpr, thr = roc_curve(yb, bench.SSS.values)
youden = float(thr[np.argmax(tpr - fpr)])
th_rows = []
for name, cut in [("fixed 0.5", 0.5), ("Youden optimum (in-sample)", youden)]:
    pred = (bench.SSS.values >= cut).astype(int)
    tp, tn = int(((pred == 1) & (yb == 1)).sum()), int(((pred == 0) & (yb == 0)).sum())
    fp, fn = int(((pred == 1) & (yb == 0)).sum()), int(((pred == 0) & (yb == 1)).sum())
    th_rows.append({"threshold": name, "value": cut, "TP": tp, "FN": fn, "TN": tn, "FP": fp,
                    "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "accuracy": (tp + tn) / len(yb)})
th = pd.DataFrame(th_rows)
save_csv(th, table("benchmark_threshold.csv"))

vus = v[(v["class"] == "VUS") & v.in_structure]
update_values("benchmark", {
    "clinvar_variants_total": int(len(v)),
    "clinvar_class_counts": counts_all,
    "binary_variants": int(len(binary)),
    "binary_in_structure": int(binary.in_structure.sum()),
    "coverage_binary": coverage,
    "benchmark_n": int(len(bench)), "benchmark_pathogenic": int(yb.sum()), "benchmark_benign": int((1 - yb).sum()),
    "performance": perf.set_index("predictor").to_dict(orient="index"),
    "delong_vs_sss": dl.set_index("comparison")[["difference", "z", "p"]].to_dict(orient="index"),
    "thresholds": th.set_index("threshold").to_dict(orient="index"),
    "vus_in_structure": int(len(vus)),
    "vus_sss_ge_0.5": int((vus.SSS >= 0.5).sum()),
})
print(perf.round(4).to_string(index=False))
print(dl.round(4).to_string(index=False))
print(th.round(3).to_string(index=False))
print("coverage among P/B:", coverage)
