# Reproduction package: Random Forest, XGBoost, and SVM for ATP match outcomes

Code to regenerate **Tables 3 to 5 and Figures 2 to 4** of the manuscript *A Comparison of Random Forest, XGBoost, and SVM in Predicting ATP Tennis Match Outcomes* from the original match records, together with the demonstration application.

## 1. Fixed inputs (fill in before archiving)

| Item | Value |
|---|---|
| Source data | Jeff Sackmann, `tennis_atp` (CC BY-NC-SA 4.0), mirror `Aneeshers/tennis-sackmann-archive` on Hugging Face, files `atp/atp_matches_2000.csv` to `atp_matches_2026.csv` |
| Dataset revision (Hugging Face commit SHA) | `[FILL: SHA found by tools/find_dataset_revision.py]` |
| Archived snapshot of the exact files used | `[FILL: Zenodo DOI]` (`data_snapshot/`, with `SHA256SUMS`) |
| Code revision | `[FILL: git commit SHA and tag, e.g. v1.0-submission]` |
| Archived release of this repository | `[FILL: Zenodo DOI of the GitHub release]` |
| Reference run environment | `[FILL: Python and package versions; the full list is in results/reproducibility/requirements_frozen.txt of the reference run]` |

The reported results come from the archived snapshot, not from the floating `main` branch of the mirror.

## 2. Run

```bash
pip install -r requirements.txt
# recommended: the archived snapshot (folder of atp_matches_YYYY.csv)
python run_all.py --data-dir ./data_snapshot --out ./outputs
# or download a pinned revision
python run_all.py --data-revision <HF_COMMIT_SHA> --out ./outputs
# smoke test with tiny budgets (NOT the reported configuration)
python tools/make_synthetic_snapshot.py ./synthetic && python run_all.py --data-dir ./synthetic --out ./out_test --quick
```

`run_all.py` refuses to run without a pinned source. Scripts are executed in order in one namespace; each is also a notebook cell, so they can be pasted into Colab in order. Expected run time on a 2-core CPU: `[FILL]` (hyperparameter searches and the SVM with Platt scaling dominate).

## 3. Where each table and figure comes from

| Manuscript item | Script | Output file(s) in `outputs/results/` |
|---|---|---|
| Table 3, search spaces and selected hyperparameters | `04_calibrate_search_space.py`, `05_nested_evaluation.py` | `protokol_dibekukan.json`, `tabel_kalibrasi_ruang.csv`, `tabel_parameter_per_lipatan.csv` |
| Table 4, accuracy, ROC-AUC, log loss, Brier, ECE with intervals | `06_table4_core_metrics.py`, `07_tables_extended.py` | `tabel_hasil_metrik_ci.csv`, `tabel_ci_bootstrap.csv`, `tabel_auc_per_tahun_ci.csv`, `tabel_baseline_probabilistik.csv`, `tabel_kalibrasi_statistik.csv` |
| Table 5, pairwise comparison (b, c, McNemar, Holm) | `07_tables_extended.py` | `tabel5_lengkap_b_c_p_holm.csv` (also `tabel_selisih_akurasi.csv`, `tabel_mcnemar.csv`) |
| Figure `[CONFIRM number]`, confusion matrices | `08_figures.py` | `fig1_confusion_matrix_en.png` |
| Figure `[CONFIRM number]`, ROC curves and calibration curves with Elo and ranking baselines | `08_figures.py` | `fig2_roc_kalibrasi_en.png` |
| Figure `[CONFIRM number]`, SHAP importance (bar and beeswarm) | `08_figures.py` | `fig3_shap_bar_en.png`, `fig4_shap_beeswarm_en.png` |

File names keep the numbering of the original notebook; the column above gives the manuscript numbering. Further analyses reported in the Results and Discussion (data dependence, imputation, subgroups, ablation, application checks) are in the notebook cells listed in `EXPERIMENT_LOG.md` and in `app/`.

## 4. Randomization and hyperparameters

All random elements, seeds and the order in which random numbers are drawn are listed in `outputs/results/reproducibility/randomization_procedures.csv` (summary: seed 42 for the player order, the hyperparameter searches, the estimators, the bootstrap, and the SHAP sample; seed 43 for the yearly-mean AUC bootstrap). The selected hyperparameters of every model in every fold are in `hyperparameters_selected_per_fold.csv` (also Table 3), the frozen search spaces in `protokol_dibekukan.json`, and the calibration of those spaces in `search_space_calibration_log.csv`.

## 5. What is archived besides the code

`scripts/09_manifest.py` writes `outputs/results/reproducibility/`:

- `reproducibility_manifest.json`: data file hashes, row counts, hashes of the raw, cleaned and feature data, package versions, git commit, script hashes, configuration, seeds, selected hyperparameters, and a fingerprint of the results (correct predictions and a hash of the rounded probabilities per model).
- `feature_matrix.csv.gz`: the 17 features for every match after the cold-start rule, with date, match counts and target.
- `predictions_test.csv.gz`: predicted probabilities and classes of the three models for every test match.
- `player_states_final.csv`, `player_surface_states_final.csv`: Elo, match counts and last-match attributes of every player after the last match in the data (the state used by the application).
- `requirements_frozen.txt` and `SHA256SUMS.txt` (every output file).

Tables 4 and 5 and Figure 3 can be recomputed from `predictions_test.csv.gz` alone, without refitting any model.

## 6. Checking a re-run against the reference run

```bash
python tools/verify_against_reference.py reference/reproducibility_manifest.json outputs/results/reproducibility/reproducibility_manifest.json
```

It checks that the data files are byte-identical, that row counts and data hashes agree, that the number of correct predictions per model is the same, and (as a warning) that rounded probabilities, selected hyperparameters and package versions agree. Probabilities may differ in the last digits across CPUs or library versions; the correct-prediction counts should not.

## 7. Application

`app/app_gradio.py` is the Gradio application with input checks, and `app/app_input_check_on_test_matches.py` applies its checks to the test matches. Run them in the same session after `scripts/05_nested_evaluation.py`. The application uses the models trained on 2010 to 2022 and the player states after the last match in the data.

## 8. Citation and licence

Cite the original data (Sackmann, J., *tennis_atp*, CC BY-NC-SA 4.0) and this repository (DOI above). The code is released under `[FILL: licence]`; the data keep their original licence.
