"""Stage 1: rebuild the residue interaction network and assemble the residue feature table.

Output: data/processed/residue_features.csv (one row per structurally resolved residue).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from gaa_hotspot import data, network
from gaa_hotspot.common import config, get_logger, processed, save_csv, update_values

log = get_logger("01_features")

typed = data.rin_edges()
g = network.build_graph(typed)
cent = network.centralities(g)
log.info("RIN: %d nodes, %d unique residue contacts, %d typed interactions", g.vcount(), g.ecount(), len(typed))

chim = data.chimera_features()
cons = data.consurf_db_scores()
labels = data.mutation_labels()
seq = data.structure_sequence()

df = cent.join(chim, how="left").join(cons, how="left").join(labels, how="left")
df.insert(0, "wt", seq.reindex(df.index))
df["mutation_observed"] = df["mutation_observed"].fillna(0).astype(int)

assert df.index.is_unique and len(df) == g.vcount()
assert set(labels.index) == set(df.index), "label file and network cover different residues"

missing = {c: int(df[c].isna().sum()) for c in config()["features"] if df[c].isna().any()}
log.info("missing feature values (imputed inside CV folds): %s", missing)

save_csv(df.reset_index(), processed("residue_features.csv"))

itype_counts = typed["type"].value_counts().to_dict()
update_values(
    "network",
    {
        "n_nodes": g.vcount(),
        "n_edges_unique_pairs": g.ecount(),
        "n_typed_interactions": len(typed),
        "typed_interactions_by_type": itype_counts,
        "connected_components": len(g.components()),
        "residue_range": [int(df.index.min()), int(df.index.max())],
        "n_residues": len(df),
        "n_mutation_positions": int(df.mutation_observed.sum()),
        "n_no_mutation": int((df.mutation_observed == 0).sum()),
        "n_zero_betweenness": int((df.betweenness == 0).sum()),
        "missing_feature_values": missing,
    },
)
log.info("residues: %d (%d mutation positions)", len(df), df.mutation_observed.sum())
