#!/usr/bin/env bash
# Reproduce every table, figure and number in results/ from data/raw/.
# Stage 3 (nested CV) takes ~20 min per run on a 10-core laptop.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
PY=${PYTHON:-python}

$PY scripts/01_build_features.py
$PY scripts/02_network_analysis.py
$PY scripts/03_train_models.py
$PY scripts/03_train_models.py --legacy811
$PY scripts/04_hotspots.py
$PY scripts/05_feature_importance.py
$PY scripts/05b_feature_ablation.py
$PY scripts/06_substitution_severity.py
$PY scripts/07_benchmark.py
$PY scripts/08_figures.py
$PY scripts/08b_fig1_workflow.py
$PY scripts/09_supplementary_tables.py
$PY scripts/10_report.py
$PY scripts/11_paper_supplement.py
$PY scripts/build_notebooks.py
$PY -m pytest -q
