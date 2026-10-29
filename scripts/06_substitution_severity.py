"""Stage 6: Substitution Severity Score for every possible substitution at every resolved residue.

Outputs
  results/tables/di_matrix.csv                 20 x 20 physicochemical distance index
  data/processed/sss_all_substitutions.csv     EVOscore, DI and SSS for every residue x substitution
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from gaa_hotspot import data, sss
from gaa_hotspot.common import AA, config, get_logger, processed, raw, read_processed, save_csv, table, update_values

log = get_logger("06_sss")
c = config()["sss"]

di, info = sss.distance_index(c["pca_variance"])
save_csv(di.reset_index().rename(columns={"index": "aa"}), table("di_matrix.csv"))
log.info("DI: %s", info)

legacy = pd.read_csv(raw("physicochemical", "legacy_DI_matrix.csv")).dropna(subset=["Reference_aa"])
L = legacy.pivot(index="Reference_aa", columns="Variety_aa", values="Physiochem_score").loc[AA, AA]
iu = np.triu_indices(20, 1)
info["pearson_r_vs_legacy_DI"] = float(np.corrcoef(di.values[iu], L.values[iu])[0, 1])

freq = data.consurf_frequencies()
features = read_processed("residue_features.csv").set_index("residue")
wt = features["wt"]
common = wt.index.intersection(freq.index)
mismatch = [int(p) for p in common if freq.loc[p, "wt_consurf"] != wt[p]]
wt_absent = [int(p) for p in common if freq.loc[p, wt[p]] == 0]
log.info("ConSurf positions %d; structure residues covered %d; wt mismatches %s; wt absent from MSA %d",
         len(freq), len(common), mismatch, len(wt_absent))

long = sss.severity_long(freq, wt.loc[common], di, c["min_frequency"], c["evoscore_cap"])
save_csv(long, processed("sss_all_substitutions.csv"))

update_values("sss", {
    **info,
    "n_positions": int(len(common)),
    "n_substitutions": int(len(long)),
    "wt_mismatch_structure_vs_consurf": mismatch,
    "n_positions_wt_absent_from_msa": len(wt_absent),
    "evoscore_range": [float(long.evoscore.min()), float(long.evoscore.max())],
    "fraction_evoscore_at_cap": float((long.evoscore >= c["evoscore_cap"] - 1e-9).mean()),
    "sss_median": float(long.sss.median()),
    "sss_iqr": [float(long.sss.quantile(0.25)), float(long.sss.quantile(0.75))],
})
print(long.describe().round(3).to_string())
