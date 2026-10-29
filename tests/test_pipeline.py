"""Consistency checks on the pipeline outputs. Run after run_all.sh: `pytest -q`."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaa_hotspot import stats  # noqa: E402
from gaa_hotspot.common import MODEL_ORDER  # noqa: E402

P, T = ROOT / "data" / "processed", ROOT / "results" / "tables"
V = json.loads((ROOT / "results" / "values.json").read_text())


def test_dataset_size():
    df = pd.read_csv(P / "residue_features.csv")
    assert len(df) == 844 and df.residue.is_unique
    assert df.mutation_observed.sum() == V["network"]["n_mutation_positions"] == 214


def test_network_counts():
    assert V["network"]["n_nodes"] == 844
    assert V["network"]["n_edges_unique_pairs"] == 3475
    assert V["network"]["connected_components"] == 1


def test_delong_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 1, 300)
    auc, lo, hi = stats.delong_ci(y, s)
    assert auc == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert lo < auc < hi


def test_oof_auc_reproducible_from_probabilities():
    prob = pd.read_csv(P / "oof_probabilities.csv")
    perf = pd.read_csv(T / "model_performance.csv").set_index("model")
    for m in MODEL_ORDER:
        assert roc_auc_score(prob.mutation_observed, prob[m]) == pytest.approx(perf.loc[m, "pooled_oof_auc"], abs=1e-10)


def test_every_residue_predicted_out_of_fold_in_every_repeat():
    long = pd.read_csv(P / "oof_predictions_long.csv")
    counts = long.groupby(["repeat", "model"]).residue.nunique()
    assert (counts == 844).all()
    assert not long.duplicated(["repeat", "model", "residue"]).any()


def test_hotspot_classes_consistent():
    v = pd.read_csv(T / "residue_votes.csv")
    vote_cols = [f"vote_{m}" for m in MODEL_ORDER]
    assert (v[vote_cols].sum(axis=1) == v.votes).all()
    novel = pd.read_csv(T / "novel_hotspots.csv")
    assert len(novel) == V["hotspots"]["n_novel"]
    assert set(novel.residue) == set(v[(v.votes >= 3) & (v.mutation_observed == 0)].residue)
    dom = pd.read_csv(T / "hotspot_domain_breakdown.csv")
    assert dom.novel_hotspots.sum() == len(novel)


def test_sss_formula_and_independence_from_eve():
    s = pd.read_csv(P / "sss_all_substitutions.csv")
    assert np.allclose(s.sss, (s.evoscore_scaled + s.di_scaled) / 2)
    assert s.evoscore_scaled.between(0, 1).all() and s.di_scaled.between(0, 1).all()
    assert not any("eve" in c.lower() for c in s.columns)


def test_benchmark_set():
    b = pd.read_csv(T / "benchmark_variants.csv")
    bench = b[b["class"].isin(["Pathogenic", "Benign"]) & b.all_four_scores]
    assert len(bench) == V["benchmark"]["benchmark_n"]
    perf = pd.read_csv(T / "benchmark_performance.csv").set_index("predictor")
    y = (bench["class"] == "Pathogenic").astype(int)
    for t in ["SSS", "AlphaMissense", "ESM-1v", "EVE"]:
        assert roc_auc_score(y, bench[t]) == pytest.approx(perf.loc[t, "roc_auc"], abs=1e-10)
