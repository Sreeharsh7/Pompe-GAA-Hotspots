"""Stage 8: main and supplementary figure panels in the style of manuscript draft v7.

Every data panel is written as a 600 dpi PNG. Panels are named after the v7
figure they replace (e.g. Fig3a_*). Protein-structure renderings (Fig. 2a, 3b, 3d, 4b) are
made in PyMOL from results/structures/ (see the .pml scripts).

Outputs
  results/figures/main/Fig3a ... Fig6
  results/figures/supplementary/FigS1 ... FigS13
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from gaa_hotspot import figures as F
from gaa_hotspot.common import MODEL_ORDER, MODEL_NAMES, config, load_values, path, read_processed, table
from gaa_hotspot.network import criticality_bin_edges
from gaa_hotspot.plotting import apply_style, save

apply_style()
cfg = config()
V = load_values()
MAIN, SUPP = path("figures_main"), path("figures_supp")
for d in (MAIN, SUPP):
    for old in d.glob("*"):
        old.unlink()

feat = read_processed("residue_features.csv").set_index("residue")
groups = read_processed("residue_network_groups.csv").set_index("residue")
votes = pd.read_csv(table("residue_votes.csv")).set_index("residue")
prob = read_processed("oof_probabilities.csv").set_index("residue")
long = read_processed("oof_predictions_long.csv")
perf = pd.read_csv(table("model_performance.csv")).set_index("model")
novel = pd.read_csv(table("novel_hotspots.csv")).set_index("residue")
sss = read_processed("sss_all_substitutions.csv")
di = pd.read_csv(table("di_matrix.csv")).set_index("aa")
bench_all = pd.read_csv(table("benchmark_variants.csv"))
bench = bench_all[bench_all["class"].isin(["Pathogenic", "Benign"]) & bench_all.all_four_scores].copy()
bperf = pd.read_csv(table("benchmark_performance.csv")).set_index("predictor")
delong = pd.read_csv(table("benchmark_delong_vs_sss.csv"))
thresholds = pd.read_csv(table("vote_thresholds.csv")).set_index("model")["threshold"].to_dict()

# ---- main figures ------------------------------------------------------------------------
corr = pd.read_csv(table("centrality_correlation.csv")).set_index("centrality")
save(F.fig2b_centrality_correlation(corr), MAIN, "Fig2b_centrality_correlation")

edges = criticality_bin_edges(feat, cfg["criticality"]["kbins"], cfg["seed"])
save(F.fig3a_criticality(feat, groups, edges), MAIN, "Fig3a_criticality_quadrants")
save(F.fig3c_supercritical_3d(feat, groups), MAIN, "Fig3c_supercritical_hubs_3D")
save(F.fig3e_venn(feat, groups), MAIN, "Fig3e_venn_mutations_HDHB_hubs")

save(F.fig4a_roc(prob), MAIN, "Fig4a_ROC_five_models")
save(F.fig4_group_profile(feat, groups, "areaSAS", "Solvent Accessible Surface Area (SASA, Å²)"), MAIN, "Fig4c_SASA_by_criticality")
save(F.fig4_group_profile(feat, groups, "conservation", "ConSurf Conservation Score (SCORE)"), MAIN, "Fig4d_conservation_by_criticality")

save(F.fig5a_schematic(di, V["sss"]["n_properties"], V["sss"]["n_components"], V["sss"]["variance_retained"]), MAIN, "Fig5a_SSS_schematic")
save(F.fig5b_novel_heatmap(sss, novel, feat), MAIN, "Fig5b_SSS_novel_hotspots")
save(F.fig5c_clinvar_heatmap(sss, bench, feat), MAIN, "Fig5c_SSS_clinvar_positions")

save(F.fig6_benchmark(bench, bperf, delong), MAIN, "Fig6_benchmark_ROC_table")

# ---- supplementary figures ---------------------------------------------------------------
fstats = pd.read_csv(table("feature_statistics.csv")).set_index("feature")
save(F.figS_feature_boxplots(feat, fstats, cfg["features"] + ["betweenness"]), SUPP, "FigS1_feature_boxplots")
save(F.figS_model_metrics(F.fold_metrics(long, cfg["ml"]["vote_threshold"])), SUPP, "FigS2_model_metrics")
save(F.figS_confusion(prob, thresholds), SUPP, "FigS3_confusion_matrices")
save(F.figS_cv_roc(long), SUPP, "FigS4_cv_ROC_mean_sd")

save(F.figS_permutation_importance(pd.read_csv(table("permutation_importance.csv")).set_index("feature")), SUPP, "FigS5a_permutation_importance")
for letter, m in zip("bcd", ["rf", "lr", "xgb"]):
    sv = read_processed(f"shap_values_{m}.csv").set_index("residue")
    xin = read_processed(f"shap_inputs_{m}.csv").set_index("residue")[sv.columns]
    save(F.figS_shap(sv, xin, f"SHAP values — {MODEL_NAMES[m]}"), SUPP, f"FigS5{letter}_SHAP_{m.upper()}")

save(F.figS_predicted_vs_clinical(votes, cfg["ml"]["consensus_min_votes"]), SUPP, "FigS6_predicted_vs_clinical")
save(F.figS_hotspot_stability(novel), SUPP, "FigS7_hotspot_stability")
title = f"Distance index: {V['sss']['n_properties']} AAindex properties, {V['sss']['n_components']} PCs ({100 * V['sss']['variance_retained']:.1f}% variance)"
save(F.figS_di_evoscore(di, sss, cfg["sss"]["evoscore_cap"], title), SUPP, "FigS8_DI_EVOscore")
save(F.figS_full_protein_sss(sss), SUPP, "FigS9_SSS_full_protein")
save(F.figS_clinvar_components(sss, bench, feat), SUPP, "FigS10_SSS_components_clinvar")
save(F.figS_benchmark_separation(bench, bperf), SUPP, "FigS11_benchmark_separation")
if table("legacy811_model_performance.csv").exists():
    save(F.figS_legacy811(perf, pd.read_csv(table("legacy811_model_performance.csv")).set_index("model")), SUPP, "FigS12_sensitivity_811")
if table("feature_ablation.csv").exists():
    save(F.figS_ablation_vote_rule(pd.read_csv(table("feature_ablation.csv")), pd.read_csv(table("vote_threshold_sensitivity.csv"))),
         SUPP, "FigS13_ablation_vote_rule")

print("figures written to", MAIN, "and", SUPP)
