"""Parsers for the raw inputs: Chimera attributes, RINerator network, ConSurf and labels."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from .common import AA, THREE_TO_ONE, config, raw

CHIMERA_ATTRS = ["areaSAS", "areaSES", "relSESA", "kdhydro", "phi", "psi"]


def read_chimera_attribute(name: str) -> pd.Series:
    """Chimera 'save attribute' output: one '/A:<resnum>\\t<value>' pair per line."""
    values = {}
    pattern = re.compile(r"/A:(\d+)\s+(\S+)")
    for line in raw("chimera", f"{name}.txt").read_text().splitlines():
        m = pattern.search(line)
        if not m:
            continue
        val = m.group(2)
        values[int(m.group(1))] = np.nan if val == "None" else float(val)
    return pd.Series(values, name=name)


def chimera_features() -> pd.DataFrame:
    df = pd.concat([read_chimera_attribute(a) for a in CHIMERA_ATTRS], axis=1)
    df.index.name = "residue"
    return df


def rin_edges() -> pd.DataFrame:
    """Unique undirected residue pairs from the RINerator SIF file.

    The SIF lists every typed interaction (contact, H-bond, overlap, combined);
    the network used here is the set of distinct residue pairs.
    """
    rows = []
    for line in raw("rin", "5NN3_relaxed_h.sif").read_text().splitlines():
        a, itype, b = line.split()
        u, v = int(a.split(":")[1]), int(b.split(":")[1])
        rows.append((min(u, v), max(u, v), itype))
    typed = pd.DataFrame(rows, columns=["u", "v", "type"])
    typed = typed[typed.u != typed.v]
    return typed


def consurf_db_scores() -> pd.Series:
    """Normalised ConSurf-DB conservation score per residue (feature used by the models)."""
    x = pd.read_excel(raw("conservation", "5NN3_relaxed_ConsurfDB_data.xlsx"))
    resnum = x["3LATOM"].astype(str).str.extract(r"[A-Z]{3}(\d+):")[0]
    keep = resnum.notna()  # positions without a structure residue are listed as '-'
    s = pd.Series(x.loc[keep, "SCORE"].values, index=resnum[keep].astype(int).values, name="conservation")
    return s[~s.index.duplicated()]


def consurf_frequencies() -> pd.DataFrame:
    """Per-position amino-acid frequencies from the ConSurf grades file (positions x 20)."""
    cfg = config()["sss"]
    rows = {}
    wt = {}
    for line in raw("conservation", "5NN3_consurf_grades.txt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 10 or not re.match(r"\s*\d+$", parts[0]):
            continue
        m = re.match(r"\s*([A-Z]{3}):(\d+):", parts[2])
        if not m:
            continue
        pos = int(m.group(2))
        freqs = {}
        variety = parts[9].strip()
        if "%" not in variety:
            # a single residue type observed at this position
            for aa in [v.strip() for v in variety.split(",") if v.strip()]:
                freqs[aa] = 1.0
        else:
            for item in variety.split(","):
                item = item.strip()
                mm = re.match(r"([A-Z])\s*(<?)\s*(\d+)%", item)
                if not mm:
                    continue
                aa, lt, pct = mm.groups()
                freqs[aa] = cfg["below_one_percent"] if lt else int(pct) / 100.0
        rows[pos] = freqs
        wt[pos] = THREE_TO_ONE.get(m.group(1), "X")
    f = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=AA).fillna(0.0)
    f.index.name = "residue"
    f.insert(0, "wt_consurf", pd.Series(wt))
    return f.sort_index()


def structure_sequence() -> pd.Series:
    chain = PDBParser(QUIET=True).get_structure("gaa", raw("structure", "5NN3_relaxed.pdb"))[0][config()["structure"]["chain"]]
    seq = {r.id[1]: THREE_TO_ONE[r.get_resname()] for r in chain if r.get_resname() in THREE_TO_ONE}
    return pd.Series(seq, name="wt").sort_index()


def mutation_labels() -> pd.Series:
    lab = pd.read_csv(raw("labels", "GAA_mutation_positions.csv"), encoding="utf-8-sig")
    return lab.set_index("node")["mutation_observed"].rename("mutation_observed")
