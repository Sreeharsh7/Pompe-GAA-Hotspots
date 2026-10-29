"""Statistical utilities: permutation tests, FDR, DeLong ROC comparison, bootstrap CIs."""

from __future__ import annotations

import numpy as np
from scipy import stats


def permutation_mean_diff(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided permutation p-value for the difference in means (with +1 correction)."""
    pooled = np.concatenate([x, y])
    nx = len(x)
    observed = abs(x.mean() - y.mean())
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:nx].mean() - pooled[nx:].mean()) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / sp


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


# Fast DeLong (Sun & Xu, IEEE Signal Process. Lett. 2014).
def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _delong_components(y: np.ndarray, scores: np.ndarray):
    """scores: (k, n). Returns AUCs (k,) and covariance (k, k)."""
    y = np.asarray(y).astype(bool)
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_ci(y: np.ndarray, score: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    auc, cov = _delong_components(y, score[None, :])
    se = np.sqrt(cov[0, 0])
    z = stats.norm.ppf(1 - alpha / 2)
    return float(auc[0]), float(max(0.0, auc[0] - z * se)), float(min(1.0, auc[0] + z * se))


def delong_test(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    aucs, cov = _delong_components(y, np.vstack([a, b]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "difference": float(aucs[0] - aucs[1]), "z": float(z), "p": float(p)}


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """Stratified percentile bootstrap 95 % CI for ROC AUC."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs.append(roc_auc_score(y[idx], score[idx]))
    return tuple(np.percentile(aucs, [2.5, 97.5]))
