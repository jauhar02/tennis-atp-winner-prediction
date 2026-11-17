# Part of the reproduction pipeline for 'A Comparison of Random Forest, XGBoost, and SVM in Predicting ATP Tennis Match Outcomes'.
# Scripts are executed in order by run_all.py in one shared namespace (they are also notebook cells: paste them in order).
# Table 4 core metrics from the pooled nested-fold probabilities (probas, preds come from 05)
def metrics(name, y, pred, proba=None, score=None):
    return {"Model": name,
            "Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred),
            "ROC-AUC": roc_auc_score(y, proba if proba is not None else score),
            "LogLoss": log_loss(y, proba) if proba is not None else np.nan,
            "Brier": brier_score_loss(y, proba) if proba is not None else np.nan}

rows = [metrics("Baseline: ranking lebih baik menang", y_test, (X_test.rank_diff_log > 0).astype(int), score=X_test.rank_diff_log),
        metrics("Baseline: Elo lebih tinggi menang", y_test, (X_test.elo_diff > 0).astype(int), score=X_test.elo_diff)]
rows += [metrics(n, y_test, preds[n], proba=probas[n]) for n in final_models]
results = pd.DataFrame(rows).set_index("Model").round(4)
results
# ===== SEL C: pengganti sel 19 dan 21 =====
display = display if "display" in globals() else print
# Interval kepercayaan memakai bootstrap BLOK (per bulan), dan uji McNemar diberi koreksi Holm.

from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

B, B_AUC = 2000, 500
blocks = test.date.dt.to_period("M").astype(str).values
ub = np.unique(blocks); idx = {b: np.where(blocks == b)[0] for b in ub}
def block_sample(rng):
    return np.concatenate([idx[b] for b in rng.choice(ub, size=len(ub), replace=True)])

rng_b = np.random.default_rng(SEED)
preds_all = {"Baseline: ranking lebih baik menang": (X_test.rank_diff_log > 0).astype(int).values,
             "Baseline: Elo lebih tinggi menang": (X_test.elo_diff > 0).astype(int).values, **preds}
acc_boot = {n: np.empty(B) for n in preds_all}
for k in range(B):
    ii = block_sample(rng_b)
    for n, p in preds_all.items():
        acc_boot[n][k] = (p[ii] == y_test[ii]).mean()
auc_boot = {n: np.empty(B_AUC) for n in probas}
for k in range(B_AUC):
    ii = block_sample(rng_b)
    for n, p in probas.items():
        auc_boot[n][k] = roc_auc_score(y_test[ii], p[ii])

def ci(x): return np.percentile(x, [2.5, 97.5])
ci_rows = []
for n in preds_all:
    lo, hi = ci(acc_boot[n])
    row = {"Model": n, "Accuracy": round(float((preds_all[n] == y_test).mean()), 4),
           "Acc CI95 bawah": round(lo, 4), "Acc CI95 atas": round(hi, 4)}
    if n in auc_boot:
        a_lo, a_hi = ci(auc_boot[n])
        row.update({"ROC-AUC": round(float(roc_auc_score(y_test, probas[n])), 4),
                    "AUC CI95 bawah": round(a_lo, 4), "AUC CI95 atas": round(a_hi, 4)})
    ci_rows.append(row)
ci_table = pd.DataFrame(ci_rows).set_index("Model")

pairs = list(combinations(list(preds), 2)) + [(m, "Baseline: Elo lebih tinggi menang") for m in preds]
diff_rows = []
for a, b in pairs:
    d = (acc_boot[a] - acc_boot[b]) * 100; lo, hi = ci(d)
    diff_rows.append({"Pasangan (A - B)": f"{a} - {b}",
                      "Selisih akurasi (poin %)": round(float((preds_all[a] == y_test).mean() - (preds_all[b] == y_test).mean()) * 100, 3),
                      "CI95 bawah": round(lo, 3), "CI95 atas": round(hi, 3), "Mencakup 0": bool(lo <= 0 <= hi)})
diff_table = pd.DataFrame(diff_rows)

correct = {n: preds[n] == y_test for n in preds}
out = []
for a, b in combinations(preds, 2):
    ca, cb = correct[a], correct[b]
    res = mcnemar([[(ca & cb).sum(), (ca & ~cb).sum()], [(~ca & cb).sum(), (~ca & ~cb).sum()]], exact=False, correction=True)
    out.append({"Pasangan": f"{a} vs {b}", "A benar, B salah": int((ca & ~cb).sum()), "A salah, B benar": int((~ca & cb).sum()),
                "Statistik": round(res.statistic, 4), "p-value": res.pvalue})
mcnemar_df = pd.DataFrame(out)
mcnemar_df["p Holm"] = multipletests(mcnemar_df["p-value"], method="holm")[1]
mcnemar_df["Signifikan (0.05, Holm)"] = mcnemar_df["p Holm"] < 0.05
mcnemar_df[["p-value", "p Holm"]] = mcnemar_df[["p-value", "p Holm"]].round(4)
results_full = results.join(ci_table.drop(columns=["Accuracy", "ROC-AUC"], errors="ignore"))
display(ci_table); display(diff_table); display(mcnemar_df)

results_full.to_csv(f"{OUT_DIR}/tabel_hasil_metrik_ci.csv"); ci_table.to_csv(f"{OUT_DIR}/tabel_ci_bootstrap.csv")
diff_table.to_csv(f"{OUT_DIR}/tabel_selisih_akurasi.csv", index=False); mcnemar_df.to_csv(f"{OUT_DIR}/tabel_mcnemar.csv", index=False)
