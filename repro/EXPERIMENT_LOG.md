# Experimental chronology and configuration log

Purpose: let a reader audit **what was decided when, on which data, and what was seen before each decision**. The manuscript states that the hyperparameter search spaces were changed after an initial evaluation on the test years; this log records that sequence and the safeguards that followed. Fill every `[FILL]` from verifiable evidence (git history, Colab and Google Drive version history, file timestamps, saved outputs), and attach the evidence file name. Do not reconstruct dates from memory.

## Summary table

| Stage | What was done | Data the decision could see | Test years seen before this stage? | Date | Evidence (commit, file, notebook revision) |
|---|---|---|---|---|---|
| A. Initial run | First pipeline: training 2010 to 2022, test 2023 to 2026, first (narrower) search spaces, one evaluation on the test years | all years | no | [FILL] | [FILL] |
| B. Search spaces widened | The best parameters of stage A lay on the edge of their spaces, so the spaces were widened (the notebook text states this) | **test results of stage A had been seen** | yes | [FILL] | [FILL] |
| C. Calibration of the spaces by a fixed rule | `04_calibrate_search_space.py`: starting from the stage B spaces, a parameter whose best value is on the edge is extended by one step (at most two rounds), using **only data before 2023** (`CAL_END`); the final spaces are frozen in `protokol_dibekukan.json` | data before 2023 only | not used by the rule | [FILL] | `search_space_calibration_log.csv`, [FILL commit] |
| D. Nested rolling-origin evaluation (reported results) | `05_nested_evaluation.py`: for each test year Y (2023 to 2026), training 2010 to Y-1, hyperparameters searched only on the latest 12,000 rows of that training window, spaces frozen from stage C | training data before Y only | not used by the procedure | [FILL] | `tabel_parameter_per_lipatan.csv`, [FILL commit] |
| E. Analyses added in response to review | Sensitivity and robustness analyses listed below; none changes the features, spaces, hyperparameters or models of stage D | stage D outputs | yes | [FILL per item] | [FILL] |

## Disclosure to include in the manuscript

The widening of the search spaces in stage B followed an evaluation on the test years. To remove this dependence, stage C recalibrated the spaces by a fixed rule on data before 2023 only, the spaces were frozen, and stage D tuned the hyperparameters of each test year on that year's past data. Analyses in stage E (seed and budget stability, tuning window, missing-value handling, dependence, baselines, calibration, subgroups, ablation, application checks) did not change any reported model.

## Stage E items (analyses in response to review)

| Item | Notebook cell or script | Output files | Changed models or hyperparameters? | Date |
|---|---|---|---|---|
| Temporal integrity test | Sel T | [FILL] | no | [FILL] |
| Data flow and exclusions | Sel U | [FILL] | no | [FILL] |
| Search budget, seed stability, tuning window | Sel V0, V1, V2 | [FILL] | no (checks only) | [FILL] |
| Preprocessing and chronological calibration | Sel W, W2 | `tabel_auc_logloss_per_lipatan.csv` | no | [FILL] |
| Player-order consistency | Sel X | [FILL] | no | [FILL] |
| Missing-value frequency and imputation sensitivity | Sel Y0, Y1 | `tabel_missing_*`, `tabel_imputasi_*` | no (retrains with the stage D parameters on altered features, for comparison only) | [FILL] |
| Dependence and inference | Sel Z, Z2 | `tabel_ketergantungan_*`, `tabel_inferensi_*`, `tabel5_lengkap_b_c_p_holm.csv` | no | [FILL] |
| Elo and ranking as probabilistic baselines | Sel AA | `tabel_baseline_*` | no | [FILL] |
| Calibration analysis | Sel AB | `tabel_kalibrasi_*` | no | [FILL] |
| Subgroup performance | Sel AC | `tabel_subkelompok_*` | no | [FILL] |
| Feature ablation | Sel AD, AD2 | `tabel_ablasi_*` | no (separate models for ablation only) | [FILL] |
| Application input checks | app cell, AppCheck | `tabel_aplikasi_pemeriksaan_laga_uji.csv` | no | [FILL] |

## Where to find evidence for the dates

- **Git history:** `git log --date=iso --format="%h %ad %s"`; tag the commit used for the reported results (for example `v1.0-submission`) and cite the tag and SHA.
- **Colab:** File > Revision history of the notebook.
- **Google Drive:** right click the checkpoint or result file > Manage versions (it shows when each version was written).
- **Output files:** the modification times of the CSV files and of `ad_scores.joblib`, `ad2_scores.joblib`.
- **Initial evaluation (stage A):** keep or recover the original outputs (accuracy and parameters) and file them in `archive/initial_run/`, with a note on which search spaces were used.
