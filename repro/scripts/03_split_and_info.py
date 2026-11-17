# Part of the reproduction pipeline for 'A Comparison of Random Forest, XGBoost, and SVM in Predicting ATP Tennis Match Outcomes'.
# Scripts are executed in order by run_all.py in one shared namespace (they are also notebook cells: paste them in order).
train = data[(data.date >= f"{TRAIN_START_YEAR}-01-01") & (data.date < TEST_START_DATE)]
test = data[data.date >= TEST_START_DATE]

X_train, y_train = train[FEATURES], train["target"].values
X_test, y_test = test[FEATURES], test["target"].values
print(f"Latih: {len(train):,} laga ({train.date.min().date()} s/d {train.date.max().date()})")
print(f"Uji  : {len(test):,} laga ({test.date.min().date()} s/d {test.date.max().date()})")

import platform, json
import importlib.metadata as md

OUT_DIR = f"{SAVE_DIR}/results"
os.makedirs(OUT_DIR, exist_ok=True)

# FIG_LANG is set in 00_setup.py ("en" = English figure labels)
L = {"id": dict(roc="Kurva ROC", calib="Kurva Kalibrasi", perfect="Sempurna", xlab="Probabilitas prediksi",
                ylab="Proporsi menang aktual", shap="Kepentingan Fitur (SHAP, XGBoost)"),
     "en": dict(roc="ROC Curve", calib="Calibration Curve", perfect="Perfect", xlab="Predicted probability",
                ylab="Observed win proportion", shap="Feature Importance (SHAP, XGBoost)")}[FIG_LANG]

def _ver(p):
    try: return md.version(p)
    except Exception: return "-"

info = {
    "Tahun data dimuat": f"{START_YEAR}-{END_YEAR} ({len(frames)} file)",
    "Baris mentah": len(raw),
    "Baris setelah pembersihan": len(df),
    f"Baris setelah aturan cold-start (>= {MIN_PRIOR_MATCHES} laga)": len(data),
    "Jumlah fitur": len(FEATURES),
    "Data latih": f"{len(train):,} laga ({train.date.min().date()} s/d {train.date.max().date()})",
    "Data uji": f"{len(test):,} laga ({test.date.min().date()} s/d {test.date.max().date()})",
    "Proporsi target=1 (latih)": round(float(train.target.mean()), 4),
    "Proporsi target=1 (uji)": round(float(test.target.mean()), 4),
    "Seed": SEED,
    "Python": platform.python_version(),
    **{p: _ver(p) for p in ["scikit-learn", "xgboost", "shap", "pandas", "numpy", "statsmodels", "gradio"]},
}
for k, v in info.items(): print(f"{k:55s}: {v}")
