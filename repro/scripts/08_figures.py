# Part of the reproduction pipeline for 'A Comparison of Random Forest, XGBoost, and SVM in Predicting ATP Tennis Match Outcomes'.
# Scripts are executed in order by run_all.py in one shared namespace (they are also notebook cells: paste them in order).
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, n in zip(axes, final_models):
    ConfusionMatrixDisplay.from_predictions(y_test, preds[n], ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix: {n}")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig1_confusion_matrix_{FIG_LANG}.png", dpi=300, bbox_inches="tight"); plt.show()


# ===== PENGGANTI BLOK GAMBAR ROC + KALIBRASI (versi R2-10: baseline Elo dan peringkat ikut digambar). Jalankan SETELAH Sel AA =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fgrid = np.linspace(0, 1, 201); fyr = test.date.dt.year.values; fys = sorted(np.unique(fyr))
def _roc_year(s, lab, **kw):
    tprs, ws, aucs = [], [], []
    for y in fys:
        m = fyr == y; fpr, tpr, _ = roc_curve(y_test[m], s[m])
        tprs.append(np.interp(fgrid, fpr, tpr)); ws.append(int(m.sum())); aucs.append(roc_auc_score(y_test[m], s[m]))
    axes[0].plot(fgrid, np.average(tprs, axis=0, weights=ws), label=f"{lab} (AUC={np.average(aucs, weights=ws):.3f})", **kw)
for n, p in probas.items():
    _roc_year(p, n)
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10)
    axes[1].plot(mean_pred, frac_pos, marker="o", label=n)
_roc_year(test.elo_diff.values, "Elo baseline", ls="--", color="gray"); _roc_year(test.rank_diff_log.values, "Ranking baseline", ls=":", color="black")
for n, c in (("Elo (expected score)", "gray"), ("Ranking (recalibrated)", "black")):
    frac_pos, mean_pred = calibration_curve(y_test, base_probs[n], n_bins=10)
    axes[1].plot(mean_pred, frac_pos, marker="s", ms=4, ls="--" if "Elo" in n else ":", color=c, label=n.replace("Elo (expected score)", "Elo baseline").replace("Ranking (recalibrated)", "Ranking baseline"))
sub = " (mean of test years)" if FIG_LANG == "en" else " (rata-rata tahun uji)"
axes[0].plot([0, 1], [0, 1], "k-", lw=.6, alpha=.4); axes[0].set(title=L["roc"] + sub, xlabel="False Positive Rate", ylabel="True Positive Rate")
axes[1].plot([0, 1], [0, 1], "k-", lw=.6, alpha=.4, label=L["perfect"]); axes[1].set(title=L["calib"], xlabel=L["xlab"], ylabel=L["ylab"])
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_roc_kalibrasi_{FIG_LANG}.png", dpi=300, bbox_inches="tight"); plt.show()

import shap
xgb_clf = final_models["XGBoost"].named_steps["clf"]
sample = X_test.sample(min(1000, len(X_test)), random_state=SEED)
shap_values = shap.TreeExplainer(xgb_clf).shap_values(sample)

shap.summary_plot(shap_values, sample, plot_type="bar", show=False); plt.title(L["shap"])
plt.savefig(f"{OUT_DIR}/fig3_shap_bar_{FIG_LANG}.png", dpi=300, bbox_inches="tight"); plt.show()
shap.summary_plot(shap_values, sample, show=False)
plt.savefig(f"{OUT_DIR}/fig4_shap_beeswarm_{FIG_LANG}.png", dpi=300, bbox_inches="tight"); plt.show()

# tabel kepentingan fitur (rata-rata |SHAP|) untuk naskah
shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=sample.columns).sort_values(ascending=False).round(4).rename("mean_abs_SHAP")
shap_imp.to_frame()
