# Part of the reproduction pipeline for 'A Comparison of Random Forest, XGBoost, and SVM in Predicting ATP Tennis Match Outcomes'.
# Scripts are executed in order by run_all.py in one shared namespace (they are also notebook cells: paste them in order).
# ===== SEL A: KALIBRASI RUANG PENCARIAN (hanya data < 2023) =====
import json, time
from sklearn.model_selection import ParameterGrid

CAL_END = "2023-01-01"            # data dari tanggal ini ke atas TIDAK boleh disentuh di sel ini
MAX_ROUNDS = 2                    # maksimum pelebaran ruang
GEO = {"clf__C", "clf__gamma", "clf__learning_rate", "clf__n_estimators"}      # diperlebar dengan rasio
LIMIT = {"clf__subsample": (0.1, 1.0), "clf__colsample_bytree": (0.1, 1.0),    # batas alami parameter
         "clf__min_child_weight": (1, None), "clf__max_depth": (1, None),
         "clf__min_samples_leaf": (1, None)}

BASE = {   # titik awal = ruang pada Tabel 3 naskah saat ini
    "Random Forest": (Pipeline([("clf", RandomForestClassifier(random_state=SEED))]),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 4, 5, 6, 8],
         "clf__min_samples_leaf": [10, 20, 40, 80], "clf__max_features": ["sqrt", 0.3, 0.5]}),
    "XGBoost": (Pipeline([("clf", XGBClassifier(eval_metric="logloss", random_state=SEED))]),
        {"clf__n_estimators": [200, 400, 800], "clf__max_depth": [2, 3, 4],
         "clf__learning_rate": [0.005, 0.01, 0.02, 0.05], "clf__subsample": [0.5, 0.7, 0.9],
         "clf__colsample_bytree": [0.5, 0.7, 0.9], "clf__min_child_weight": [1, 5, 10, 20]}),
    "SVM": (Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=SEED))]),
        {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1], "clf__gamma": [0.001, 0.003, 0.01, 0.03, "scale"]}),
}

def extend_space(grid, best):
    """Perlebar satu langkah pada parameter numerik yang terbaiknya di tepi ruang. Aturan tetap, tanpa data uji."""
    new, edges = {k: list(v) for k, v in grid.items()}, []
    for k, vals in grid.items():
        nums = sorted(v for v in vals if isinstance(v, (int, float)))
        if len(nums) < 3 or best[k] not in (nums[0], nums[-1]):
            continue
        low = best[k] == nums[0]
        if k in GEO:
            step = (nums[1] / nums[0]) if low else (nums[-1] / nums[-2])
            cand = nums[0] / step if low else nums[-1] * step
        else:
            step = (nums[1] - nums[0]) if low else (nums[-1] - nums[-2])
            cand = nums[0] - step if low else nums[-1] + step
        lo, hi = LIMIT.get(k, (None, None))
        if (lo is not None and cand < lo) or (hi is not None and cand > hi):
            continue                                   # sudah di batas alami, tidak diperlebar
        cand = int(round(cand)) if all(isinstance(v, int) for v in nums) else round(float(cand), 6)
        if cand not in new[k]:
            new[k].append(cand); edges.append(k.replace("clf__", ""))
    return new, edges

def tune(pipe, grid, X, y):
    n_iter = min(N_ITER, len(ParameterGrid(grid)))
    s = RandomizedSearchCV(pipe, grid, n_iter=n_iter, cv=TimeSeriesSplit(n_splits=3), scoring="roc_auc",
                           n_jobs=-1, random_state=SEED, refit=False)
    s.fit(X, y)
    return s.best_params_, s.best_score_, n_iter

dev = data[(data.date >= f"{TRAIN_START_YEAR}-01-01") & (data.date < CAL_END)]
assert dev.date.max() < pd.Timestamp(CAL_END)
X_dev, y_dev = dev[FEATURES].tail(N_TUNE), dev["target"].values[-N_TUNE:]

FROZEN, cal_log = {}, []
for name, (pipe, grid) in BASE.items():
    g = grid
    for rnd in range(MAX_ROUNDS + 1):
        bp, sc, n_iter = tune(pipe, g, X_dev, y_dev)
        g2, edges = extend_space(g, bp)
        cal_log.append({"Model": name, "Putaran": rnd, "Ukuran ruang": len(ParameterGrid(g)),
                        "ROC-AUC CV": round(sc, 4), "Parameter di tepi": ", ".join(edges) or "-"})
        print(f"{name:14s} putaran {rnd} | ruang {len(ParameterGrid(g)):5d} | AUC {sc:.4f} | di tepi: {edges or '-'}")
        if not edges or rnd == MAX_ROUNDS:
            break
        g = g2
    FROZEN[name] = (pipe, g)                          # ruang final = ruang terakhir yang dipakai

pd.DataFrame(cal_log).to_csv(f"{OUT_DIR}/tabel_kalibrasi_ruang.csv", index=False)
with open(f"{OUT_DIR}/protokol_dibekukan.json", "w") as f:
    json.dump({"kalibrasi_data_sampai": CAL_END, "N_TUNE": N_TUNE, "N_ITER": N_ITER, "MAX_ROUNDS": MAX_ROUNDS,
               "ruang": {k: {pk: [str(x) for x in pv] for pk, pv in g.items()} for k, (_, g) in FROZEN.items()}},
              f, indent=2)
print("\nRUANG DIBEKUKAN. Mulai sel B, jangan ubah FROZEN lagi.")
