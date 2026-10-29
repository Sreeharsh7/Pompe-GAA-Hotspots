"""Stage 4: model votes, consensus hotspots, domain distribution and active-site proximity.

Outputs
  results/tables/residue_votes.csv            per-residue OOF probabilities, votes and class
  results/tables/novel_hotspots.csv           novel consensus hotspots with annotations
  results/tables/hotspot_domain_breakdown.csv
  results/tables/discordant_positions.csv     mutation positions with zero votes vs consensus positions
  results/structures/*.pdb, *.pml             structures with votes / criticality in the B-factor column
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, NeighborSearch, PDBParser
from scipy import stats as ss

from gaa_hotspot.common import MODEL_ORDER, ROOT, config, domain_of, get_logger, raw, read_processed, save_csv, table, update_values

log = get_logger("04_hotspots")
cfg = config()
spec, kmin = cfg["ml"]["vote_specificity"], cfg["ml"]["consensus_min_votes"]

prob = read_processed("oof_probabilities.csv").set_index("residue")
feat = read_processed("residue_features.csv").set_index("residue")
groups = read_processed("residue_network_groups.csv").set_index("residue")
long = read_processed("oof_predictions_long.csv")



def spec_threshold(score: pd.Series, y: pd.Series) -> float:
    """Probability at which the model reaches the target specificity on unannotated residues."""
    return float(np.quantile(score[y == 0], spec))


y = prob["mutation_observed"]
thresholds = {m: spec_threshold(prob[m], y) for m in MODEL_ORDER}
votes = pd.DataFrame({m: (prob[m] >= thresholds[m]).astype(int) for m in MODEL_ORDER})
thr_rows = [{"model": m, "threshold": thresholds[m], "sensitivity": float(votes.loc[y == 1, m].mean()),
             "specificity": float(1 - votes.loc[y == 0, m].mean()), "residues_flagged": int(votes[m].sum())} for m in MODEL_ORDER]
save_csv(pd.DataFrame(thr_rows), table("vote_thresholds.csv"))

# Sensitivity of the hotspot set to the vote rule
from sklearn.metrics import roc_curve


def rule_threshold(m, rule):
    s = prob[m]
    if rule == "probability 0.5":
        return 0.5
    if rule == "Youden optimum":
        f, t, th = roc_curve(y, s)
        return th[np.argmax(t - f)]
    if rule == "prevalence-matched":
        return np.quantile(s, 1 - y.mean())
    return np.quantile(s[y == 0], float(rule.split()[0]) / 100)


sens_rows = []
for rule in ["probability 0.5", "Youden optimum", "prevalence-matched", "85 % specificity", "90 % specificity", "95 % specificity"]:
    vv = pd.DataFrame({m: (prob[m] >= rule_threshold(m, rule)).astype(int) for m in MODEL_ORDER}).sum(axis=1)
    cons_ = vv >= kmin
    sens_rows.append({"vote_rule": rule, "flagged_by_at_least_one_model": int((vv >= 1).sum()), "consensus": int(cons_.sum()),
                      "consensus_known_positions": int((cons_ & (y == 1)).sum()), "novel_consensus_hotspots": int((cons_ & (y == 0)).sum()),
                      "novel_with_5_votes": int((cons_ & (y == 0) & (vv == 5)).sum())})
save_csv(pd.DataFrame(sens_rows), table("vote_threshold_sensitivity.csv"))
df = feat[["wt", "mutation_observed", "areaSAS", "conservation", "degree", "betweenness", "closeness"]].join(prob[MODEL_ORDER].add_prefix("prob_"))
df = df.join(votes.add_prefix("vote_"))
df["votes"] = votes.sum(axis=1)
df["mean_probability"] = prob[MODEL_ORDER].mean(axis=1)
df["consensus"] = df.votes >= kmin

# How often a residue reaches consensus in each independent CV repeat (stability)
rep_votes = []
for (r, m), d in long.groupby(["repeat", "model"]):
    t = np.quantile(d.loc[d.y == 0, "probability"], spec)
    rep_votes.append(d.assign(vote=(d.probability >= t).astype(int)))
rep = pd.concat(rep_votes).groupby(["repeat", "residue"]).vote.sum().unstack(0)
df["consensus_repeat_fraction"] = (rep >= kmin).mean(axis=1).reindex(df.index)


def klass(r):
    if r.consensus and r.mutation_observed:
        return "Consensus, known mutation position"
    if r.consensus:
        return "Novel consensus hotspot"
    if r.mutation_observed:
        return "Known mutation position, below consensus"
    return "Not prioritised"


df["class"] = df.apply(klass, axis=1)
df["domain"] = [domain_of(p) for p in df.index]
df = df.join(groups[["criticality", "supercritical"]])

# Active-site proximity (minimum heavy-atom distance) -------------------------------------
asite = cfg["active_site"]
site_role = {r: "nucleophile" for r in asite["nucleophile"]} | {r: "acid/base" for r in asite["acid_base"]} | {r: "substrate binding" for r in asite["substrate_binding"]}
chain = PDBParser(QUIET=True).get_structure("gaa", raw("structure", "5NN3_relaxed.pdb"))[0][cfg["structure"]["chain"]]
res = {r.id[1]: r for r in chain}
heavy = lambda r: [a for a in r if a.element != "H"]


def min_dist(a, b):
    xa = np.array([at.coord for at in heavy(res[a])])
    xb = np.array([at.coord for at in heavy(res[b])])
    return float(np.sqrt(((xa[:, None] - xb[None]) ** 2).sum(-1)).min())


near, near_d = [], []
for p in df.index:
    best = min(((min_dist(p, s), s) for s in site_role if s != p), key=lambda t: t[0])
    near.append(f"{res[best[1]].get_resname().title()}{best[1]} ({site_role[best[1]]})")
    near_d.append(best[0])
df["nearest_active_site_residue"] = near
df["nearest_active_site_distance_A"] = near_d
df["is_active_site_residue"] = df.index.isin(list(site_role))
save_csv(df.reset_index(), table("residue_votes.csv"))

novel = df[df["class"] == "Novel consensus hotspot"].copy()
novel["residue_label"] = [f"{w}{p}" for p, w in novel.wt.items()]
cols = ["residue_label", "wt", "domain", "votes"] + [f"vote_{m}" for m in MODEL_ORDER] + ["mean_probability", "consensus_repeat_fraction",
        "criticality", "supercritical", "areaSAS", "conservation", "is_active_site_residue", "nearest_active_site_residue", "nearest_active_site_distance_A"]
save_csv(novel[cols].reset_index(), table("novel_hotspots.csv"))

# Domain distribution and enrichment relative to domain size -------------------------------
n_all, n_nov = len(df), len(novel)
dom_rows = []
for name, a, b in cfg["domains"]:
    size = int(df.index.to_series().between(a, b).sum())
    k = int(novel.index.to_series().between(a, b).sum())
    exp = n_nov * size / n_all
    p = ss.binomtest(k, n_nov, size / n_all).pvalue if n_nov else np.nan
    dom_rows.append({"domain": name, "range": f"{a}-{b}", "residues_in_structure": size, "novel_hotspots": k,
                     "percent_of_novel": 100 * k / n_nov if n_nov else np.nan, "expected_by_size": exp, "binomial_p": p})
dom = pd.DataFrame(dom_rows)
save_csv(dom, table("hotspot_domain_breakdown.csv"))

# Discordant known positions ---------------------------------------------------------------
known = df[df.mutation_observed == 1]
zero = known[known.votes == 0]
cons_known = known[known.consensus]
disc_rows = []
for f in ["areaSAS", "conservation", "degree", "betweenness", "closeness"]:
    disc_rows.append({"feature": f, "median_zero_vote_positions": zero[f].median(), "median_consensus_positions": cons_known[f].median(),
                      "mann_whitney_p": ss.mannwhitneyu(zero[f], cons_known[f]).pvalue})
disc = pd.DataFrame(disc_rows)
save_csv(disc, table("discordant_positions.csv"))

# Overlap with the hotspot set reported in manuscript draft v7 ------------------------------
V7 = [92, 135, 136, 244, 247, 258, 280, 282, 296, 302, 315, 317, 331, 338, 344, 353, 354, 364, 365, 370, 378, 399, 402, 407, 443,
      488, 516, 518, 597, 606, 639, 673, 683, 703, 710, 720, 722, 747, 754, 821, 833, 951]
shared = sorted(set(V7) & set(novel.index))

# Annotated structures for PyMOL / ChimeraX rendering ---------------------------------------
out = ROOT / "results" / "structures"
out.mkdir(parents=True, exist_ok=True)
crit_code = {"Other": 0, "LDLB": 1, "LDHB": 2, "HDHB": 3}
for tag, values in [("votes", df.votes), ("criticality", df.criticality.map(crit_code) + 1 * df.supercritical)]:
    s = PDBParser(QUIET=True).get_structure("gaa", raw("structure", "5NN3_relaxed.pdb"))
    for r in s[0][cfg["structure"]["chain"]]:
        val = float(values.get(r.id[1], -1)) if r.id[0] == " " or r.id[0] == "H_CSO" else -1.0
        for a in r:
            a.set_bfactor(val)
    io = PDBIO()
    io.set_structure(s)
    io.save(str(out / f"5NN3_{tag}_in_bfactor.pdb"))
(out / "render_votes.pml").write_text(
    "load 5NN3_votes_in_bfactor.pdb, gaa\nremove hydro\nhide everything\nshow cartoon, polymer\nbg_color white\n"
    "spectrum b, blue_white_red, polymer and b > -0.5, minimum=0, maximum=5\nset cartoon_transparency, 0\n"
    f"show sticks, polymer and resi {'+'.join(map(str, novel.index))} and not name N+C+O\n"
    "set ray_opaque_background, 0\norient\nray 2400, 2400\npng fig4b_votes.png, dpi=600\n")
(out / "render_criticality.pml").write_text(
    "load 5NN3_criticality_in_bfactor.pdb, gaa\nremove hydro\nhide everything\nshow cartoon, polymer\nbg_color white\ncolor grey80, polymer\n"
    "color 0x0072B2, b > 0.5 and b < 1.5\ncolor 0xD55E00, b > 1.5 and b < 2.5\ncolor 0x009E73, b > 2.5 and b < 3.5\n"
    "color 0xE69F00, b > 3.5\nshow spheres, b > 0.5 and name CA\nset sphere_scale, 0.9\norient\nray 2400, 2400\npng fig3b_criticality.png, dpi=600\n")

update_values("hotspots", {
    "vote_rule": f"mean OOF probability >= model-specific {int(spec * 100)} % specificity threshold",
    "vote_thresholds": thresholds, "consensus_min_votes": kmin,
    "vote_threshold_sensitivity": sens_rows,
    "n_flagged_by_at_least_one_model": int((df.votes >= 1).sum()),
    "n_consensus": int(df.consensus.sum()),
    "n_consensus_known": int(len(cons_known)),
    "n_novel": int(n_nov),
    "n_known_below_consensus": int((known.votes < kmin).sum()),
    "n_known_zero_votes": int(len(zero)),
    "consensus_recall_of_known_positions": float(len(cons_known) / len(known)),
    "novel_vote_distribution": novel.votes.value_counts().sort_index().to_dict(),
    "novel_residues": novel.residue_label.tolist(),
    "novel_consensus_in_all_repeats": int((novel.consensus_repeat_fraction == 1).sum()),
    "novel_active_site_residues": novel[novel.is_active_site_residue].residue_label.tolist(),
    "novel_within_8A_of_active_site": int((novel.nearest_active_site_distance_A <= 8).sum()),
    "domain_counts": dom.set_index("domain").novel_hotspots.to_dict(),
    "v7_42_hotspots_reproduced": len(shared),
    "v7_42_hotspots_shared": shared,
    "discordant": disc.set_index("feature").to_dict(orient="index"),
})
log.info("consensus %d (known %d, novel %d); flagged by >=1 model %d; shared with v7 list %d/42",
         df.consensus.sum(), len(cons_known), n_nov, (df.votes >= 1).sum(), len(shared))
print(novel[cols].round(3).to_string())
print(dom.round(3).to_string(index=False))
