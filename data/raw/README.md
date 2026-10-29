# Raw inputs

Everything the pipeline uses. Nothing in this folder is modified by the scripts.

| File | Content | Source |
|---|---|---|
| `structure/5NN3_relaxed.pdb` | Human GAA (chain A, residues 81–952) after Rosetta relaxation | PDB [5NN3](https://www.rcsb.org/structure/5NN3) (Roig-Zamboni et al., *Nat. Commun.* 2017) |
| `chimera/*.txt` | Per-residue areaSAS, areaSES, relSESA, Kyte–Doolittle hydrophobicity, φ, ψ | UCSF Chimera 1.15 on the relaxed model |
| `rin/5NN3_relaxed_h.sif`, `*.ea` | Residue interaction network (typed interactions, interaction counts/scores) | RINerator 0.5.1 after adding hydrogens |
| `conservation/5NN3_relaxed_ConsurfDB_data.xlsx` | Normalised ConSurf-DB conservation score per residue (model feature) | [ConSurf-DB](https://consurfdb.tau.ac.il) |
| `conservation/5NN3_consurf_grades.txt` | ConSurf grades file with per-position amino-acid frequencies (used for EVOscore) | [ConSurf](https://consurf.tau.ac.il) run on 5NN3 |
| `labels/GAA_mutation_positions.csv` | 1 = residue with a reported missense mutation, 0 otherwise (844 residues) | Curated from HGMD (Stenson et al., 2020) |
| `physicochemical/aaindex1_seqinr.csv` | 544 AAindex1 property scales for the 20 amino acids | R package SeqinR 4.2-36 (`data(aaindex)`), exported by `scripts/00_export_aaindex.R` |
| `physicochemical/legacy_DI_matrix.csv` | DI matrix used in earlier drafts (comparison only) | Earlier analysis |
| `predictors/AlphaMissense_P10253.tsv` | AlphaMissense scores for GAA | [AlphaMissense](https://alphamissense.hegelab.org) (Cheng et al., *Science* 2023) |
| `predictors/ESM1v_P10253.csv` | ESM-1v log-likelihood ratios for GAA | Meier et al., 2021 |
| `figure_icons/*.png` | Icons used in the Fig. 1 workflow schematic | Drawn for this study |
| `predictors/EVE_LYAG_HUMAN.csv` | EVE scores and the bundled ClinVar annotations used as benchmark labels | [EVE](https://evemodel.org) (Frazer et al., *Nature* 2021) |

## Licensing notes before making the repository public

- **HGMD**: the label file contains only binary residue-level flags, not HGMD records. Check that this is allowed under your HGMD licence (HGMD Professional terms restrict redistribution). If in doubt, replace it with ClinVar-derived positions or provide it on request.
- **AlphaMissense**, **EVE** and **ESM-1v** score files: confirm the licence of the release you downloaded before redistributing them (otherwise ship a download script instead). **ClinVar** is public domain.
- ConSurf, AAindex and PDB data are freely available with citation.
