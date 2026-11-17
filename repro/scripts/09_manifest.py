# Part of the reproduction pipeline for 'A Comparison of Random Forest, XGBoost, and SVM in Predicting ATP Tennis Match Outcomes'.
# Scripts are executed in order by run_all.py in one shared namespace (they are also notebook cells: paste them in order).
# ===== 09_manifest: reproducibility manifest, portable exports, and checksums (R2-15) =====
# Run last. Needs everything produced by 00-08 (raw, df, data, tr, FROZEN, fold_params, cal_log, probas, preds, test, file_hashes).
import subprocess, datetime, importlib.metadata as md

REPRO = f"{OUT_DIR}/reproducibility"; os.makedirs(REPRO, exist_ok=True)
_sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
_h = lambda d: int(pd.util.hash_pandas_object(d, index=False).values.sum(dtype=np.uint64))
def _git(*a):
    try: return subprocess.run(["git", *a], cwd=globals().get("SCRIPTS_DIR", "."), capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception: return None

# ---------- 1. data snapshot ----------
hf_head = None
if not DATA_DIR and not RAW_FILE:
    try:
        import requests
        hf_head = requests.get(f"https://huggingface.co/api/datasets/{HF_REPO}", timeout=20).json().get("sha")
    except Exception: pass
M = {"created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
     "data": {"source_repo": f"https://huggingface.co/datasets/{HF_REPO}", "upstream": "https://github.com/JeffSackmann/tennis_atp (CC BY-NC-SA 4.0)",
              "revision_requested": DATA_REVISION or None, "local_snapshot_dir": DATA_DIR or None, "raw_file": RAW_FILE or None,
              "revision_pinned": bool(DATA_REVISION or DATA_DIR or RAW_FILE), "hf_head_sha_when_run": hf_head,
              "files": file_hashes, "n_files": len(file_hashes), "rows_raw": int(len(raw)), "rows_after_cleaning": int(len(df)),
              "rows_after_cold_start": int(len(data)), "hash_raw_rows": _h(raw), "hash_cleaned_rows": _h(df.drop(columns=["round_order"], errors="ignore")),
              "hash_feature_matrix": _h(data[FEATURES + ["target"]])}}
if not M["data"]["revision_pinned"]: print("WARNING: the data revision is not pinned; the manifest records file hashes but not a revision.")

# ---------- 2. software environment and code ----------
_dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions() if d.metadata["Name"])
open(f"{REPRO}/requirements_frozen.txt", "w").write("\n".join(_dists) + "\n")
_keys = ["scikit-learn", "xgboost", "shap", "pandas", "numpy", "scipy", "statsmodels", "matplotlib", "joblib", "gradio"]
_ver = {}
for k in _keys:
    try: _ver[k] = md.version(k)
    except Exception: _ver[k] = None
_sd = globals().get("SCRIPTS_DIR")
M["environment"] = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
                    "key_packages": _ver, "n_installed_packages": len(_dists), "colab": "google.colab" in sys.modules,
                    "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
                    "script_sha256": {f: _sha(os.path.join(_sd, f)) for f in sorted(os.listdir(_sd)) if f.endswith(".py")} if _sd else None,
                    "quick_mode": QUICK}

# ---------- 3. configuration and feature definitions ----------
M["configuration"] = {"SEED": SEED, "START_YEAR": START_YEAR, "END_YEAR": END_YEAR, "TRAIN_START_YEAR": TRAIN_START_YEAR, "TEST_START_DATE": TEST_START_DATE,
                      "OUTER_YEARS": OUTER_YEARS, "MIN_PRIOR_MATCHES": MIN_PRIOR_MATCHES, "DROP_RETIRED": DROP_RETIRED, "N_TUNE": N_TUNE, "N_ITER": N_ITER,
                      "FEATURES": FEATURES, "initial_elo": 1500, "elo_k_rule": "K = 250 / (n_matches + 5) ** 0.4 (general and per surface)",
                      "form_window_matches": 10, "fatigue_window_days": 14, "missing_minutes_imputed": 100, "surface_win_rate_prior": "(wins + 2.5) / (n + 5)",
                      "elapsed_unit_for_fatigue": "hours (minutes / 60)"}

# ---------- 4. randomization procedures ----------
rz = pd.DataFrame([
    ["Player order (Player 1 / Player 2)", "np.random.default_rng(42), one rng.random() < 0.5 per row of the cleaned match list in chronological order (rows later removed by the cold-start rule still consume a draw)", "02_features.py"],
    ["Hyperparameter search", "RandomizedSearchCV(random_state=42, n_iter=min(60, grid size), TimeSeriesSplit(n_splits=3), scoring=roc_auc, refit=False) on the latest N_TUNE=12,000 rows of each training window", "04_calibrate_search_space.py, 05_nested_evaluation.py"],
    ["Estimators", "random_state=42 for RandomForestClassifier, XGBClassifier and SVC (SVC probability=True uses Platt scaling with internal 5-fold CV)", "04, 05"],
    ["Parallelism", "n_jobs=-1 for Random Forest, XGBoost and the search; results are expected to be identical for the same library versions, small floating-point differences are possible across CPUs", "04, 05"],
    ["Accuracy and AUC confidence intervals", "np.random.default_rng(42); calendar-month block bootstrap; B=2000 accuracy resamples followed by B_AUC=500 AUC resamples from the same generator", "06_table4_core_metrics.py"],
    ["Yearly-mean AUC intervals", "np.random.default_rng(43) (SEED+1); calendar-month blocks; 1,000 resamples", "07_tables_extended.py (C2)"],
    ["Baseline and calibration intervals", "np.random.default_rng(42); calendar-month blocks", "07_tables_extended.py (AA, AB)"],
    ["SHAP sample", "X_test.sample(1000, random_state=42)", "08_figures.py"],
    ["Data splits", "none random: chronological; training 2010 to year-1, test year Y for Y in 2023..2026 (nested rolling origin)", "05_nested_evaluation.py"]],
    columns=["Component", "Seed and procedure", "Script"])
rz.to_csv(f"{REPRO}/randomization_procedures.csv", index=False); M["randomization"] = rz.to_dict("records")

# ---------- 5. hyperparameters, search spaces and calibration log ----------
pd.DataFrame(fold_params).to_csv(f"{REPRO}/hyperparameters_selected_per_fold.csv", index=False)
pd.DataFrame(cal_log).to_csv(f"{REPRO}/search_space_calibration_log.csv", index=False)
M["search_spaces_frozen"] = {k: {pk: [str(x) for x in pv] for pk, pv in g.items()} for k, (_, g) in FROZEN.items()}
M["hyperparameters_selected_per_fold"] = pd.DataFrame(fold_params).astype(str).to_dict("records")

# ---------- 6. portable exports (readable without pickles): features, predictions, final player states ----------
data[["date", "n1", "n2", "target"] + FEATURES].to_csv(f"{REPRO}/feature_matrix.csv.gz", index=False)
_pt = test[["date", "target"]].copy(); _pt["fold_year"] = test.date.dt.year.values
for n in probas: _pt[f"proba_{n}"] = probas[n]; _pt[f"pred_{n}"] = preds[n]
_pt["elo_diff"] = X_test.elo_diff.values; _pt["rank_diff_log"] = X_test.rank_diff_log.values
_pt.to_csv(f"{REPRO}/predictions_test.csv.gz", index=False)
pd.DataFrame([{"player_id": p, "name": tr.bio[p]["name"], "matches": tr.n[p], "elo": round(tr.elo[p], 4), "last_match": tr.bio[p]["last_date"].date(),
               "rank": tr.bio[p]["rank"], "age_at_last_match": tr.bio[p]["age"], "height": tr.bio[p]["ht"], "hand": tr.bio[p]["hand"]}
              for p in tr.bio]).to_csv(f"{REPRO}/player_states_final.csv", index=False)
pd.DataFrame([{"player_id": k[0], "surface": k[1], "elo_surface": round(v, 4), "matches_on_surface": tr.n_s[k], "wins_on_surface": tr.wins_s[k]}
              for k, v in tr.elo_s.items()]).to_csv(f"{REPRO}/player_surface_states_final.csv", index=False)

# ---------- 7. result fingerprint for verification across machines ----------
_ya = test.target.values
M["result_fingerprint"] = {"n_test": int(len(_ya)),
                           **{n: {"correct_predictions": int((preds[n] == _ya).sum()), "accuracy": round(float((preds[n] == _ya).mean()), 6),
                                  "sha256_probabilities_6dp": hashlib.sha256(np.round(probas[n], 6).astype("<f8").tobytes()).hexdigest()} for n in probas}}
print("\nResult fingerprint:"); [print(f"  {n:14s} correct {v['correct_predictions']}/{len(_ya)}  accuracy {v['accuracy']}") for n, v in M["result_fingerprint"].items() if n != "n_test"]

# ---------- 8. checksums of every output and the manifest ----------
_files = sorted(os.path.relpath(os.path.join(a, f), OUT_DIR) for a, _, fs in os.walk(OUT_DIR) for f in fs if f not in ("SHA256SUMS.txt", "reproducibility_manifest.json"))
M["output_sha256"] = {f: _sha(os.path.join(OUT_DIR, f)) for f in _files}
open(f"{OUT_DIR}/SHA256SUMS.txt", "w").write("".join(f"{h}  {f}\n" for f, h in M["output_sha256"].items()))
json.dump(M, open(f"{REPRO}/reproducibility_manifest.json", "w"), indent=2, default=str)
print(f"\nManifest: {REPRO}/reproducibility_manifest.json | {len(_files)} output files checksummed in {OUT_DIR}/SHA256SUMS.txt")
