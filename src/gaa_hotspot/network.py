"""Residue interaction network centralities and topological criticality groups."""

from __future__ import annotations

import igraph as ig
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import KBinsDiscretizer

CENTRALITIES = ["degree", "betweenness", "closeness", "burt", "authority", "page_rank", "k_core"]


def build_graph(edges: pd.DataFrame) -> ig.Graph:
    pairs = edges[["u", "v"]].drop_duplicates()
    nodes = sorted(set(pairs.u) | set(pairs.v))
    g = ig.Graph()
    g.add_vertices([str(n) for n in nodes])
    g.add_edges([(str(u), str(v)) for u, v in pairs.itertuples(index=False)])
    return g


def centralities(g: ig.Graph) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "degree": g.degree(),
            "betweenness": g.betweenness(),
            "closeness": g.closeness(),
            "burt": g.constraint(),
            "authority": g.authority_score(),
            "page_rank": g.pagerank(),
            "k_core": g.coreness(),
        },
        index=pd.Index([int(n) for n in g.vs["name"]], name="residue"),
    )
    df["log_betweenness"] = np.log10(1.0 + df["betweenness"])
    return df.sort_index()


def criticality_groups(df: pd.DataFrame, n_bins: int, seed: int) -> pd.Series:
    """HDHB / LDHB / LDLB from K-means bins of degree and log betweenness.

    Lowest bin = 'low', highest bin = 'high'; residues in intermediate bins are 'Other'.
    """
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans", random_state=seed)
    bins = kbd.fit_transform(df[["log_betweenness", "degree"]])
    lb, deg = bins[:, 0], bins[:, 1]
    top = n_bins - 1
    group = np.full(len(df), "Other", dtype=object)
    group[(deg == 0) & (lb == 0)] = "LDLB"
    group[(deg == top) & (lb == top)] = "HDHB"
    group[(deg == 0) & (lb == top)] = "LDHB"
    return pd.Series(group, index=df.index, name="criticality")


def criticality_bin_edges(df: pd.DataFrame, n_bins: int, seed: int) -> dict:
    """K-means bin edges used by criticality_groups (for the dashed lines of Fig. 3a)."""
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans", random_state=seed)
    kbd.fit(df[["log_betweenness", "degree"]])
    return {"log_betweenness": np.sort(kbd.bin_edges_[0]), "degree": np.sort(kbd.bin_edges_[1])}


def supercritical(df: pd.DataFrame, n: int, standardise: bool) -> pd.Series:
    """Euclidean distance from the ideal point (maximum of degree, log betweenness, closeness).

    With standardise=False the raw scales are used (as in the original analysis), which
    makes degree dominate the distance; standardise=True z-scores each axis first.
    """
    x = df[["degree", "log_betweenness", "closeness"]].astype(float)
    if standardise:
        x = (x - x.mean()) / x.std(ddof=0)
    ideal = x.max().values.reshape(1, -1)
    d = cdist(x.values, ideal).ravel()
    return pd.Series(d, index=df.index, name="distance_to_ideal")
