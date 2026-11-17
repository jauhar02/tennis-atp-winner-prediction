# Part of the reproduction pipeline for 'A Comparison of Random Forest, XGBoost, and SVM in Predicting ATP Tennis Match Outcomes'.
# Scripts are executed in order by run_all.py in one shared namespace (they are also notebook cells: paste them in order).
# ===== SEL B: EVALUASI ROLLING-ORIGIN BERSARANG (ruang FROZEN, tuning hanya di data latih tiap lipatan) =====
OUTER_YEARS = [2023, 2024, 2025, 2026]
test = data[data.date >= TEST_START_DATE].reset_index(drop=True)          # urutan kronologis, sama seperti sebelumnya
X_test, y_test = test[FEATURES], test["target"].values

probas = {n: np.full(len(test), np.nan) for n in FROZEN}
fold_rows, fold_params, models_fold1 = [], [], None
for Y in OUTER_YEARS:
    tr_f = data[(data.date >= f"{TRAIN_START_YEAR}-01-01") & (data.date < f"{Y}-01-01")]
    te_mask = (test.date.dt.year == Y).values
    if te_mask.sum() == 0:
        continue
    Xtr, ytr = tr_f[FEATURES], tr_f["target"].values
    assert tr_f.date.max() < test.date[te_mask].min()                      # latih selalu lebih lama dari uji
    Xtu, ytu = Xtr.tail(N_TUNE), ytr[-N_TUNE:]
    fitted = {}
    for name, (pipe, grid) in FROZEN.items():
        t0 = time.perf_counter()
        bp, sc, _ = tune(pipe, grid, Xtu, ytu)                             # hanya melihat data < Y
        m = clone(pipe).set_params(**bp)
        m.set_params(clf__probability=True) if name == "SVM" else m.set_params(clf__n_jobs=-1)
        m.fit(Xtr, ytr)
        probas[name][te_mask] = m.predict_proba(X_test[te_mask])[:, 1]
        fitted[name] = m
        fold_params.append({"Tahun uji": Y, "Model": name, "CV AUC": round(sc, 4), "Parameter": str(bp),
                            "Waktu (s)": round(time.perf_counter() - t0, 1)})
        acc = ((probas[name][te_mask] >= 0.5).astype(int) == y_test[te_mask]).mean()
        fold_rows.append({"Tahun uji": Y, "n": int(te_mask.sum()), "Model": name, "Akurasi": round(acc, 4)})
        print(f"{Y} | {name:13s} | akurasi {acc:.4f} | {time.perf_counter()-t0:.0f}s")
    if models_fold1 is None:
        models_fold1 = fitted                                              # latih 2010-2022 (untuk SHAP dan aplikasi)

assert not any(np.isnan(p).any() for p in probas.values())
preds = {n: (p >= 0.5).astype(int) for n, p in probas.items()}
final_models = models_fold1
fold_table = pd.DataFrame(fold_rows).pivot(index=["Tahun uji", "n"], columns="Model", values="Akurasi")
fold_table["Elo"] = [((X_test.elo_diff[(test.date.dt.year == y).values] > 0).astype(int).values ==
                      y_test[(test.date.dt.year == y).values]).mean().round(4) for y, _ in fold_table.index]
display(fold_table); fold_table.to_csv(f"{OUT_DIR}/tabel_akurasi_per_tahun.csv")
pd.DataFrame(fold_params).to_csv(f"{OUT_DIR}/tabel_parameter_per_lipatan.csv", index=False)
