# GAA structural hotspot analysis

Code and data for **"Uncovering novel pathogenic hotspots in the Pompe disease target using graph-based features and ensemble machine learning"** (Sreeharsh K, Manjari Kiran, Mohd. Akif; University of Hyderabad).

The pipeline maps human lysosomal acid α-glucosidase (GAA; PDB 5NN3) onto a residue interaction network (RIN), trains five classifiers on network, structural and conservation features to prioritise residue positions associated with reported Pompe disease missense mutations, and scores individual substitutions with a Substitution Severity Score (SSS) that is benchmarked against AlphaMissense, ESM-1v and EVE.

Every number, table and figure in `results/` is regenerated from `data/raw/` by `run_all.sh`.

## Key results

| | |
|---|---|
| Residues analysed | 844 (214 reported mutation positions) |
| RIN | 844 nodes, 3,475 residue–residue contacts |
| Criticality groups | 39 HDHB, 4 LDHB, 45 LDLB; 10 supercritical hubs (6 are mutation positions) |
| Classifier ROC AUC (pooled out-of-fold, 5 × 10-fold nested CV) | LR 0.802, SVM 0.796, RF 0.789, XGB 0.786, kNN 0.770 |
| Consensus hotspots (≥ 3/5 models, 90 % specificity per model) | 144 = 92 known positions + **52 novel** |
| SSS benchmark (102 ClinVar variants: 89 P/LP, 13 B/LB) | SSS 0.953, AlphaMissense 0.969, ESM-1v 0.947, EVE 0.936 (no significant differences) |

The full set of numbers, with confidence intervals and tests, is in [`results/RESULTS_REPORT.md`](results/RESULTS_REPORT.md).

## Repository layout

```text
config.yaml                 all analysis settings (seed, CV design, thresholds, domains)
data/raw/                   inputs (structure, Chimera, RINerator, ConSurf, labels, AAindex, predictors, Fig. 1 icons) — see data/raw/README.md
data/processed/             intermediate tables written by the pipeline
src/gaa_hotspot/            library code (parsers, network, ML, statistics, SSS, plotting style)
scripts/                    numbered pipeline stages 00–11 and the notebook builder
notebooks/                  executed step-by-step notebooks 01–05 (RIN → models → hotspots → SSS → benchmark)
results/
  RESULTS_REPORT.md         every value quoted in the manuscript
  values.json               the same values, machine-readable
  tables/                   result tables (CSV)
  figures/main/             Fig. 1 (workflow), 2b, 3a,c,e · 4a,c,d · 5a–c · 6 (600 dpi PNG)
  figures/supplementary/    additional analyses, FigS1–S13 (600 dpi PNG; repository numbering, see below)
  supplementary/paper/      Supplementary Information as published: Figs. S1–S5 (PDF) and Tables S1–S4 (Excel)
  supplementary/            Supplementary_Tables.xlsx with all 25 result tables (repository numbering) + CSV copies
  structures/               PDB files with votes / criticality in the B-factor column + PyMOL scripts
tests/                      consistency checks on the outputs
```

## Reproducing the analysis

```bash
conda env create -f environment.yml      # Python 3.12 + R/SeqinR
conda activate gaa-hotspot
pip install -e .
./run_all.sh                             # ~45 min on a 10-core laptop
```

`run_all.sh` runs the stages in order:

| Stage | Script | What it does |
|---|---|---|
| 0 | `00_export_aaindex.R` | export the 544 AAindex properties from SeqinR (output already in `data/raw/`) |
| 1 | `01_build_features.py` | rebuild the RIN from the RINerator SIF, compute centralities, merge Chimera and ConSurf features |
| 2 | `02_network_analysis.py` | HDHB/LDHB/LDLB groups, supercritical hubs, univariate permutation tests |
| 3 | `03_train_models.py` | repeated nested CV of LR, RF, SVM, XGBoost and kNN (`--legacy811` for the sensitivity run) |
| 4 | `04_hotspots.py` | votes, consensus and novel hotspots, domain enrichment, active-site distances, annotated PDBs |
| 5 | `05_feature_importance.py`, `05b_feature_ablation.py` | permutation importance, SHAP, feature-group ablation |
| 6 | `06_substitution_severity.py` | DI matrix, EVOscore and SSS for all 16,036 possible substitutions |
| 7 | `07_benchmark.py` | ClinVar benchmark against AlphaMissense, ESM-1v and EVE |
| 8–10 | `08_figures.py`, `08b_fig1_workflow.py`, `09_supplementary_tables.py`, `10_report.py` | figures, workflow schematic, full results workbook, results report |
| 11 | `11_paper_supplement.py` | the paper's Supplementary Information (Figs. S1–S5, Tables S1–S4) |

Then `pytest -q` checks that the reported numbers can be recomputed from the stored predictions.

## Supplementary material in the paper

The paper cites five supplementary figures and four supplementary tables, numbered in order of citation and built by `scripts/11_paper_supplement.py` into `results/supplementary/paper/`. The repository keeps a larger set of figures and tables with its own numbering; they correspond as follows.

| Paper | Content | Repository file |
|---|---|---|
| Fig. S1 | Classifier metrics (accuracy, F1, AUC, balanced accuracy) | `figures/supplementary/FigS2_model_metrics.png` |
| Fig. S2 | Confusion matrices at 0.5 and at 90 % specificity | `figures/supplementary/FigS3_confusion_matrices.png` |
| Fig. S3 | Cross-validated ROC curves (mean ± SD) | `figures/supplementary/FigS4_cv_ROC_mean_sd.png` |
| Fig. S4 | Permutation importance (a) and SHAP for RF, LR, XGBoost (b–d) | `figures/supplementary/FigS5a–d_*.png` |
| Fig. S5 | Stability of the 52 novel hotspots across CV repeats | `figures/supplementary/FigS7_hotspot_stability.png` |
| Table S1 | Univariate feature comparison | `tables/feature_statistics.csv` |
| Table S2 | Classifier performance | `tables/model_performance.csv` |
| Table S3 | The 52 novel consensus hotspots | `tables/novel_hotspots.csv` |
| Table S4 | ClinVar variants with SSS, AlphaMissense, ESM-1v and EVE scores | `tables/benchmark_variants.csv` |

## Method summary

- **Features (14).** RIN centralities (degree, log10(1 + betweenness), closeness, Burt's constraint, authority, PageRank, k-core) computed with python-igraph on the RINerator network of the Rosetta-relaxed 5NN3 model; Chimera structural attributes (SASA, SESA, relative SESA, Kyte–Doolittle hydrophobicity, φ, ψ); ConSurf-DB conservation score.
- **Models.** Logistic regression, random forest, SVM, XGBoost and kNN in an imblearn pipeline (median imputation → standardisation → ADASYN → classifier). Hyperparameters were tuned by inner 5-fold grid search on ROC AUC inside each outer fold of a 5 × repeated stratified 10-fold CV, so every residue is scored by models that never saw it.
- **Consensus hotspots.** A model votes for a residue when its mean out-of-fold probability reaches that model's 90 %-specificity threshold. Residues with ≥ 3 votes are consensus hotspots; those without a reported mutation are *novel*.
- **SSS.** EVOscore = −log10(f_variant / f_wildtype) from ConSurf alignment frequencies (floor 0.1 %), clipped to [0, 3] and divided by 3. DI = Euclidean distance between amino acids in the first 18 principal components (99.4 % variance) of 544 standardised AAindex properties, divided by its maximum. SSS = (EVOscore + DI) / 2. SSS uses no EVE information.
- **Benchmark.** ClinVar P/LP versus B/LB GAA missense variants (snapshot distributed with the EVE model), restricted to variants scored by all four predictors; DeLong and bootstrap CIs, paired DeLong tests.

## Data and licences

Code: MIT (see `LICENSE`). Third-party inputs keep their own terms — read `data/raw/README.md` before making the repository public, in particular regarding the HGMD-derived position labels.

## Citation

See `CITATION.cff`. Citation details will be updated on publication.
