"""Substitution Severity Score: evolutionary constraint (EVOscore) + physicochemical distance (DI).

EVOscore(i->j) = -log10(f_j / f_i) from ConSurf alignment frequencies at the position.
DI(i, j)       = Euclidean distance between amino acids i and j in the PCA space of the
                 544 AAindex properties distributed with SeqinR (PCs retaining 99 % variance).
SSS(i->j)      = mean of the two terms after scaling each to [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .common import AA, raw


def distance_index(variance: float) -> tuple[pd.DataFrame, dict]:
    ai = pd.read_csv(raw("physicochemical", "aaindex1_seqinr.csv"))
    props = ai[AA]
    n_missing = int(props.isna().any(axis=1).sum())
    props = props.apply(lambda r: r.fillna(r.mean()), axis=1)  # 13 indices miss one value
    z = StandardScaler().fit_transform(props.T.values)  # 20 amino acids x 544 properties
    pca = PCA().fit(z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, variance) + 1)
    d = squareform(pdist(pca.transform(z)[:, :k]))
    di = pd.DataFrame(d, index=AA, columns=AA)
    info = {"n_properties": int(len(props)), "n_properties_imputed": n_missing, "n_components": k, "variance_retained": float(cum[k - 1]), "di_max": float(d.max())}
    return di, info


def evoscore(freq: pd.DataFrame, wt: pd.Series, floor: float) -> pd.DataFrame:
    """EVOscore for every position (rows) and substituted residue (columns)."""
    f = freq[AA].clip(lower=floor)
    f_wt = np.array([f.loc[p, wt[p]] for p in f.index])
    return -np.log10(f.div(f_wt, axis=0))


def severity_long(freq: pd.DataFrame, wt: pd.Series, di: pd.DataFrame, floor: float, cap: float) -> pd.DataFrame:
    positions = [p for p in wt.index if p in freq.index]
    wt = wt.loc[positions]
    evo = evoscore(freq.loc[positions], wt, floor)
    di_max = di.values.max()
    rows = []
    for p in positions:
        w = wt[p]
        for aa in AA:
            if aa == w:
                continue
            e = float(evo.loc[p, aa])
            e_n = min(max(e, 0.0), cap) / cap
            d_n = float(di.loc[w, aa]) / di_max
            rows.append({"residue": p, "wt": w, "mt": aa, "variant": f"{w}{p}{aa}", "f_wt": float(freq.loc[p, w]), "f_mt": float(freq.loc[p, aa]),
                         "evoscore": e, "evoscore_scaled": e_n, "di": float(di.loc[w, aa]), "di_scaled": d_n, "sss": (e_n + d_n) / 2})
    return pd.DataFrame(rows)
