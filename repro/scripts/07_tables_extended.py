# Part of the reproduction pipeline for 'A Comparison of Random Forest, XGBoost, and SVM in Predicting ATP Tennis Match Outcomes'.
# Scripts are executed in order by run_all.py in one shared namespace (they are also notebook cells: paste them in order).
# ===== SEL C2: ROC-AUC MENURUT TAHUN UJI (rata-rata berbobot) DENGAN INTERVAL KEPERCAYAAN BOOTSTRAP BLOK (R2-5) =====
# Alasan: model tiap lipatan berbeda dan skala probabilitasnya berbeda; AUC gabungan lintas lipatan mencampur skala tersebut.
from IPython.display import display

_yr = test.date.dt.year.values; _years = [y for y in OUTER_YEARS if (_yr == y).any()]
_blk = test.date.dt.to_period("M").astype(str).values; _ub = np.unique(_blk); _idx = {k: np.where(_blk == k)[0] for k in _ub}
SKOR = {**{n: probas[n] for n in probas}, "Baseline Elo": X_test.elo_diff.values, "Baseline peringkat": X_test.rank_diff_log.values}
NB = 1000

def auc_tahun(ii):
    """Rata-rata AUC per tahun uji, berbobot jumlah laga tahun itu pada sampel ii."""
    out = {n: 0.0 for n in SKOR}; tot = 0
    for y in _years:
        sel = ii[_yr[ii] == y]
        if len(sel) == 0 or len(np.unique(y_test[sel])) < 2: continue
        for n, s in SKOR.items(): out[n] += len(sel) * roc_auc_score(y_test[sel], s[sel])
        tot += len(sel)
    return {n: v / tot for n, v in out.items()}

full = auc_tahun(np.arange(len(y_test)))
rg = np.random.default_rng(SEED + 1); boot = {n: np.empty(NB) for n in SKOR}
for k in range(NB):
    ii = np.concatenate([_idx[b_] for b_ in rg.choice(_ub, size=len(_ub), replace=True)])
    r = auc_tahun(ii)
    for n in SKOR: boot[n][k] = r[n]
t_auc = pd.DataFrame({n: {"AUC rata-rata per tahun": round(full[n], 4), "CI95 bawah": round(np.percentile(boot[n], 2.5), 4),
                          "CI95 atas": round(np.percentile(boot[n], 97.5), 4)} for n in SKOR}).T
display(t_auc)

pairs = [("Random Forest", "Baseline Elo"), ("XGBoost", "Baseline Elo"), ("SVM", "Baseline Elo"),
         ("SVM", "Random Forest"), ("SVM", "XGBoost"), ("Random Forest", "XGBoost")]
rows = []
for a, b_ in pairs:
    d = boot[a] - boot[b_]; lo, hi = np.percentile(d, [2.5, 97.5])
    rows.append({"Pasangan (A - B)": f"{a} - {b_}", "Selisih AUC": round(full[a] - full[b_], 4), "CI95 bawah": round(lo, 4),
                 "CI95 atas": round(hi, 4), "Mencakup 0": bool(lo <= 0 <= hi)})
t_auc_d = pd.DataFrame(rows); display(t_auc_d)

# AUC per tahun: model dikurangi Elo (konsistensi arah di setiap tahun uji)
per = []
for y in _years:
    m_ = _yr == y; e = roc_auc_score(y_test[m_], SKOR["Baseline Elo"][m_])
    per.append({"Tahun uji": y, **{f"{n} - Elo": round(roc_auc_score(y_test[m_], probas[n][m_]) - e, 4) for n in probas}})
t_per = pd.DataFrame(per).set_index("Tahun uji"); display(t_per)
t_auc.to_csv(f"{OUT_DIR}/tabel_auc_per_tahun_ci.csv"); t_auc_d.to_csv(f"{OUT_DIR}/tabel_auc_per_tahun_selisih.csv"); t_per.to_csv(f"{OUT_DIR}/tabel_auc_vs_elo_per_tahun.csv")


# ===== SEL W2: AUC, LOG LOSS DAN BRIER PER LIPATAN (R2-5): memisahkan kemampuan peringkat dari efek penggabungan skala probabilitas antar-lipatan =====
from IPython.display import display

rows = []
for Y in OUTER_YEARS:
    mk = (test.date.dt.year == Y).values; yy = y_test[mk]
    r = {"Tahun uji": Y, "n": int(mk.sum())}
    for n in probas: r[f"AUC {n}"] = roc_auc_score(yy, probas[n][mk])
    r["AUC Elo"] = roc_auc_score(yy, X_test.elo_diff.values[mk]); r["AUC peringkat"] = roc_auc_score(yy, X_test.rank_diff_log.values[mk])
    for n in probas: r[f"LogLoss {n}"] = log_loss(yy, probas[n][mk])
    for n in probas: r[f"Brier {n}"] = brier_score_loss(yy, probas[n][mk])
    rows.append(r)
t = pd.DataFrame(rows).set_index("Tahun uji")
w = t["n"].values; rata = (t.drop(columns="n").mul(w, axis=0)).sum() / w.sum()
t_fold = pd.concat([t.drop(columns="n").round(4), pd.DataFrame({"rata-rata berbobot n": rata.round(4)}).T])
display(t_fold.T)
t_fold.to_csv(f"{OUT_DIR}/tabel_auc_logloss_per_lipatan.csv")


# ===== SEL Z2: TABEL 5 LENGKAP (R2-9): selisih, laga tidak sepakat (b, c), uji McNemar dan penyesuaian Holm untuk SEMUA pasangan =====
# Butuh test, y_test, preds (setelah sel evaluasi). Konvensi sama dengan Tabel 5: b = A benar dan B salah; c = A salah dan B benar; selisih = (b - c) / n.
from itertools import combinations
from scipy import stats
from IPython.display import display

_yy = np.asarray(y_test); _nn = len(_yy)
_ok2 = {m: (np.asarray(preds[m]) == _yy) for m in preds}
_ok2["Elo baseline"] = ((test.elo_diff.values > 0).astype(int) == _yy)          # baseline Elo: pemain ber-Elo lebih tinggi menang
_pr2 = list(combinations(list(preds), 2)) + [(m, "Elo baseline") for m in preds]

def _holm2(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj

_z2 = []
for a, b in _pr2:
    n_b = int((_ok2[a] & ~_ok2[b]).sum()); n_c = int((~_ok2[a] & _ok2[b]).sum())
    chi2 = (abs(n_b - n_c) - 1) ** 2 / (n_b + n_c)
    _z2.append({"Pair (A - B)": f"{a} - {b}", "Difference (pts)": round(100 * (n_b - n_c) / _nn, 2),
                "b (A right, B wrong)": n_b, "c (A wrong, B right)": n_c, "Both right": int((_ok2[a] & _ok2[b]).sum()),
                "Both wrong": int((~_ok2[a] & ~_ok2[b]).sum()), "Disagreements": n_b + n_c,
                "McNemar chi2 (cc)": round(chi2, 2), "p (McNemar)": float(stats.chi2.sf(chi2, 1)),
                "p (exact binomial)": float(stats.binomtest(n_b, n_b + n_c, 0.5).pvalue)})
t_z2 = pd.DataFrame(_z2)
for fam_name, ix in (("model pairs", [0, 1, 2]), ("against Elo", [3, 4, 5])):
    t_z2.loc[ix, "Holm (McNemar)"] = _holm2(t_z2.loc[ix, "p (McNemar)"].values)
    t_z2.loc[ix, "Holm (exact)"] = _holm2(t_z2.loc[ix, "p (exact binomial)"].values)
t_z2["Holm family"] = ["model pairs"] * 3 + ["against Elo"] * 3
for c in ("p (McNemar)", "p (exact binomial)", "Holm (McNemar)", "Holm (exact)"): t_z2[c] = t_z2[c].round(4)
display(t_z2)
print("Kolom selang kepercayaan 95% pada Tabel 5 tidak diubah (bootstrap blok bulan, sudah ada).")
t_z2.to_csv(f"{OUT_DIR}/tabel5_lengkap_b_c_p_holm.csv", index=False)


# ===== SEL AA: ELO DAN PERINGKAT SEBAGAI PREDIKTOR PROBABILISTIK (R2-10). Butuh data, test, y_test, probas (setelah sel evaluasi) =====
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from IPython.display import display

OUTER_YEARS = [2023, 2024, 2025, 2026]
AAB = 1000                                                    # resampel bootstrap (blok bulan kalender)
_ty = test.date.dt.year.values; _ya = np.asarray(y_test); _na = len(_ya)

# ---------- probabilitas baseline ----------
# (1) Elo: skor harapan Elo, 1 / (1 + 10^(-selisih/400)). Ini probabilitas yang didefinisikan oleh aturan Elo itu sendiri (tanpa penyesuaian apa pun).
# (2) Elo dikalibrasi ulang: regresi logistik satu fitur (elo_diff) pada tahun latih tiap lipatan (< tahun uji), seperti model lain.
# (3) Peringkat: peringkat tidak memiliki probabilitas bawaan, maka regresi logistik satu fitur (rank_diff_log) pada tahun latih tiap lipatan.
_ec = np.full(_na, np.nan); _rc = np.full(_na, np.nan)
for _Y in OUTER_YEARS:
    _trm = ((data.date >= f"{TRAIN_START_YEAR}-01-01") & (data.date < f"{_Y}-01-01")).values; _tm = (_ty == _Y)
    for _col, _st in (("elo_diff", _ec), ("rank_diff_log", _rc)):
        _lr = LogisticRegression(C=1e9, max_iter=1000).fit(data.loc[_trm, [_col]].values, data.target.values[_trm])
        _st[_tm] = _lr.predict_proba(test.loc[_tm, [_col]].values)[:, 1]
assert not np.isnan(_ec).any() and not np.isnan(_rc).any()
base_probs = {"Elo (expected score)": 1 / (1 + 10 ** (-test.elo_diff.values / 400)),
              "Elo (recalibrated)": _ec, "Ranking (recalibrated)": _rc}
_allp = {**base_probs, **{n: np.asarray(p) for n, p in probas.items()}}
_cont = {"Elo (expected score)": test.elo_diff.values, "Elo (recalibrated)": test.elo_diff.values, "Ranking (recalibrated)": test.rank_diff_log.values}

def _wauc_a(s):
    a, w = [], []
    for yy in OUTER_YEARS:
        k = _ty == yy
        if k.sum() and len(np.unique(_ya[k])) == 2: a.append(roc_auc_score(_ya[k], s[k])); w.append(int(k.sum()))
    return float(np.average(a, weights=w))
def _ece_a(p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for k in range(bins):
        m = b == k
        if m.any(): e += m.mean() * abs(p[m].mean() - _ya[m].mean())
    return e
_ll = lambda p: -(_ya * np.log(np.clip(p, 1e-15, 1)) + (1 - _ya) * np.log(np.clip(1 - p, 1e-15, 1)))
_br = lambda p: (p - _ya) ** 2

# ---------- tabel utama ----------
rows = []
for n_, p_ in _allp.items():
    rows.append({"Predictor": n_, "Accuracy": round(float(((p_ >= .5).astype(int) == _ya).mean()), 4),
                 "ROC-AUC (yearly mean)": round(_wauc_a(_cont.get(n_, p_)), 4), "Log loss": round(float(log_loss(_ya, p_)), 4),
                 "Brier": round(float(brier_score_loss(_ya, p_)), 4), "ECE (10 bins)": round(float(_ece_a(p_)), 4)})
t_base = pd.DataFrame(rows).set_index("Predictor"); display(t_base)
print("Cek: AUC dari skor kontinu (selisih Elo / peringkat) sama dengan AUC dari probabilitas baseline:",
      all(abs(_wauc_a(_cont[k]) - _wauc_a(base_probs[k])) < 1e-9 for k in base_probs))
print("Akurasi baseline berbasis tanda selisih (definisi Tabel 4): Elo", round(float(((test.elo_diff.values > 0).astype(int) == _ya).mean()), 4),
      "| peringkat", round(float(((test.rank_diff_log.values > 0).astype(int) == _ya).mean()), 4))

# ---------- per tahun uji ----------
rows = []
for yy in OUTER_YEARS:
    k = _ty == yy
    for n_, p_ in _allp.items():
        rows.append({"Test year": yy, "Predictor": n_, "n": int(k.sum()), "Log loss": round(float(log_loss(_ya[k], p_[k])), 4), "Brier": round(float(brier_score_loss(_ya[k], p_[k])), 4)})
t_yr = pd.DataFrame(rows).pivot(index="Predictor", columns="Test year", values=["Log loss", "Brier"]); display(t_yr)

# ---------- selisih berpasangan (model - baseline), bootstrap blok bulan ----------
_mo = pd.factorize(test.date.dt.to_period("M"))[0]; _o = np.argsort(_mo, kind="stable"); _bd = np.cumsum(np.r_[0, np.bincount(_mo)])
_bl = [_o[_bd[i]:_bd[i + 1]] for i in range(len(_bd) - 1)]; _rg = np.random.default_rng(SEED)
_ix = [np.concatenate([_bl[g] for g in _rg.integers(0, len(_bl), len(_bl))]) for _ in range(AAB)]
rows = []
for m_ in probas:
    for b_ in base_probs:
        for met, fn in (("Log loss", _ll), ("Brier", _br)):
            d_ = fn(_allp[m_]) - fn(base_probs[b_]); bs = np.array([d_[i].mean() for i in _ix])
            rows.append({"Model": m_, "Baseline": b_, "Metric": met, "Model - baseline": round(float(d_.mean()), 4),
                         "95% CI": f"{np.percentile(bs, 2.5):.4f} to {np.percentile(bs, 97.5):.4f}"})
t_dlt = pd.DataFrame(rows); display(t_dlt)

# ---------- data kurva kalibrasi (10 kelas lebar sama) ----------
rows = []
for n_, p_ in _allp.items():
    b = np.minimum((p_ * 10).astype(int), 9)
    for k in range(10):
        m = b == k
        if m.any(): rows.append({"Predictor": n_, "Bin": f"{k / 10:.1f}-{(k + 1) / 10:.1f}", "n": int(m.sum()), "Mean predicted": round(float(p_[m].mean()), 4), "Observed win rate": round(float(_ya[m].mean()), 4)})
t_bins = pd.DataFrame(rows)
t_base.to_csv(f"{OUT_DIR}/tabel_baseline_probabilistik.csv"); t_yr.to_csv(f"{OUT_DIR}/tabel_baseline_per_tahun.csv")
t_dlt.to_csv(f"{OUT_DIR}/tabel_baseline_selisih_berpasangan.csv", index=False); t_bins.to_csv(f"{OUT_DIR}/tabel_baseline_kalibrasi_bins.csv", index=False)


# ===== SEL AB: ANALISIS KALIBRASI LENGKAP (R2-11). Butuh test, y_test, probas; base_probs dari Sel AA bila ada. Jalankan setelah Sel AA =====
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from IPython.display import display

OUTER_YEARS = [2023, 2024, 2025, 2026]
ABB = 1000                                                                # resampel bootstrap (blok bulan kalender)
NBIN = 10                                                                 # jumlah kelas probabilitas
_ya = np.asarray(y_test); _ty = test.date.dt.year.values; _na = len(_ya)
_PR = {n: np.asarray(p) for n, p in probas.items()}
if "base_probs" in globals():
    _PR.update({k: np.asarray(v) for k, v in base_probs.items()})
_MODELS = list(probas)

def _logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def _calfit(p, y):                                                         # kalibrasi linear: logit(P(y=1)) = a + b * logit(p)
    lr = LogisticRegression(C=1e9, max_iter=1000).fit(_logit(p)[:, None], y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])
def _citl(p, y):                                                           # 'calibration-in-the-large': intercept dengan kemiringan dikunci = 1 (Newton)
    z = _logit(p); a = 0.0
    for _ in range(25):
        q = 1 / (1 + np.exp(-(a + z))); a += (y - q).sum() / max((q * (1 - q)).sum(), 1e-12)
    return float(a)
def _wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n; c = (ph + z * z / (2 * n)) / d; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
def _ece_u(p, y, bins=NBIN):
    b = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for k in range(bins):
        m = b == k
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e
def _ece_q(p, y, bins=NBIN):                                               # kelas berisi jumlah laga sama (kuantil)
    o = np.argsort(p, kind="stable"); e = 0.0
    for ix in np.array_split(o, bins): e += len(ix) / len(p) * abs(p[ix].mean() - y[ix].mean())
    return e
def _murphy(p, y, bins=NBIN):
    b = np.minimum((p * bins).astype(int), bins - 1); rel = res = 0.0; ob = y.mean()
    for k in range(bins):
        m = b == k
        if m.any(): rel += m.mean() * (p[m].mean() - y[m].mean()) ** 2; res += m.mean() * (y[m].mean() - ob) ** 2
    return rel, res, ob * (1 - ob)
def _wauc_b(p):
    a, w = [], []
    for yy in OUTER_YEARS:
        k = _ty == yy
        if k.sum() and len(np.unique(_ya[k])) == 2: a.append(roc_auc_score(_ya[k], p[k])); w.append(int(k.sum()))
    return float(np.average(a, weights=w))

# ---------- 1. tabel kelas (prosedur: 10 kelas berlebar sama pada [0,1]; kelas k memuat p dalam [k/10, (k+1)/10), kelas terakhir tertutup) ----------
_abr = []
for n_, p_ in _PR.items():
    b = np.minimum((p_ * NBIN).astype(int), NBIN - 1)
    for k in range(NBIN):
        m = b == k
        if not m.any(): continue
        kk, nn = int(_ya[m].sum()), int(m.sum()); lo, hi = _wilson(kk, nn)
        _abr.append({"Predictor": n_, "Bin": f"{k / NBIN:.1f}-{(k + 1) / NBIN:.1f}", "n": nn, "Mean predicted": round(float(p_[m].mean()), 4),
                     "Observed frequency": round(kk / nn, 4), "95% CI (Wilson)": f"{lo:.3f} to {hi:.3f}",
                     "Gap (observed - predicted)": round(kk / nn - float(p_[m].mean()), 4), "CI covers predicted": bool(lo <= p_[m].mean() <= hi)})
t_cbin = pd.DataFrame(_abr)
display(t_cbin[t_cbin.Predictor.isin(_MODELS)])
sumr = []
for n_ in _MODELS:
    d = t_cbin[t_cbin.Predictor == n_]
    sumr.append({"Model": n_, "Bins used": len(d), "Min n": int(d.n.min()), "Max n": int(d.n.max()), "Bins with n < 100": int((d.n < 100).sum()),
                 "Share of matches in bins 0.2-0.8 (%)": round(100 * float(d[d.Bin.isin(["0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8"])].n.sum() / _na), 1),
                 "Bins whose CI misses the diagonal": int((~d["CI covers predicted"]).sum())})
t_csum = pd.DataFrame(sumr).set_index("Model"); display(t_csum)

# ---------- 2. statistik kalibrasi, dengan selang bootstrap blok bulan ----------
_mo = pd.factorize(test.date.dt.to_period("M"))[0]; _o = np.argsort(_mo, kind="stable"); _bd = np.cumsum(np.r_[0, np.bincount(_mo)])
_bl = [_o[_bd[i]:_bd[i + 1]] for i in range(len(_bd) - 1)]; _rg = np.random.default_rng(SEED)
_ixs = [np.concatenate([_bl[g] for g in _rg.integers(0, len(_bl), len(_bl))]) for _ in range(ABB)]
_abr = []
for n_, p_ in _PR.items():
    a, s = _calfit(p_, _ya); c0 = _citl(p_, _ya); rel, res, unc = _murphy(p_, _ya)
    bs = np.array([[_ece_u(p_[i], _ya[i]), *_calfit(p_[i], _ya[i]), _citl(p_[i], _ya[i])] for i in _ixs])
    ll = float(log_loss(_ya, p_)); zz = _logit(p_)[:, None]
    lr = LogisticRegression(C=1e9, max_iter=1000).fit(zz, _ya); ll_re = float(log_loss(_ya, lr.predict_proba(zz)[:, 1]))
    _fmt = lambda j, d=3: f"{np.percentile(bs[:, j], 2.5):.{d}f} to {np.percentile(bs[:, j], 97.5):.{d}f}"
    _abr.append({"Predictor": n_, "ROC-AUC (yearly mean)": round(_wauc_b(p_), 4), "ECE, 10 equal-width bins": round(_ece_u(p_, _ya), 4), "ECE 95% CI": _fmt(0, 4),
                 "ECE, 10 equal-count bins": round(_ece_q(p_, _ya), 4), "Calibration slope": round(s, 3), "Slope 95% CI": _fmt(2),
                 "Calibration intercept": round(a, 3), "Intercept 95% CI": _fmt(1), "Calibration-in-the-large": round(c0, 3), "CITL 95% CI": _fmt(3),
                 "Reliability (Murphy)": round(rel, 5), "Resolution (Murphy)": round(res, 5), "Log loss": round(ll, 4),
                 "Log loss after in-sample linear recalibration": round(ll_re, 4), "Calibration loss (log loss)": round(ll - ll_re, 4)})
t_cal = pd.DataFrame(_abr).set_index("Predictor"); display(t_cal.T)
print("Kemiringan < 1: terlalu yakin (overconfident); > 1: kurang yakin. Intersep dan CITL > 0: peluang pemain 1 menang diremehkan.")

# ---------- 3. per tahun uji: kemiringan, intersep dan ECE ----------
_abr = []
for n_, p_ in _PR.items():
    for yy in OUTER_YEARS:
        k = _ty == yy; a, s = _calfit(p_[k], _ya[k])
        _abr.append({"Predictor": n_, "Test year": yy, "n": int(k.sum()), "Slope": round(s, 3), "Intercept": round(a, 3), "ECE": round(_ece_u(p_[k], _ya[k]), 4)})
t_cyr = pd.DataFrame(_abr); display(t_cyr[t_cyr.Predictor.isin(_MODELS)].pivot(index="Predictor", columns="Test year", values=["Slope", "ECE"]))

t_cbin.to_csv(f"{OUT_DIR}/tabel_kalibrasi_kelas_wilson.csv", index=False); t_csum.to_csv(f"{OUT_DIR}/tabel_kalibrasi_ringkasan_kelas.csv")
t_cal.to_csv(f"{OUT_DIR}/tabel_kalibrasi_statistik.csv"); t_cyr.to_csv(f"{OUT_DIR}/tabel_kalibrasi_per_tahun.csv", index=False)

# ---------- 4. gambar tambahan (untuk repositori): kurva keandalan dengan selang Wilson dan histogram jumlah laga per kelas ----------
fig, axs = plt.subplots(2, len(_MODELS), figsize=(4.6 * len(_MODELS), 6.2), gridspec_kw={"height_ratios": [3, 1.2]}, sharex=True)
for j, n_ in enumerate(_MODELS):
    d = t_cbin[t_cbin.Predictor == n_]; lo = [float(s.split(" to ")[0]) for s in d["95% CI (Wilson)"]]; hi = [float(s.split(" to ")[1]) for s in d["95% CI (Wilson)"]]
    axs[0, j].errorbar(d["Mean predicted"], d["Observed frequency"], yerr=[d["Observed frequency"] - lo, np.array(hi) - d["Observed frequency"]], fmt="o-", capsize=3)
    axs[0, j].plot([0, 1], [0, 1], "k--", lw=.8); axs[0, j].set(title=n_, ylabel="Observed win proportion" if j == 0 else "", xlim=(0, 1), ylim=(0, 1))
    axs[1, j].bar(d["Mean predicted"], d["n"], width=.07); axs[1, j].set(xlabel="Predicted probability", ylabel="Matches per bin" if j == 0 else "")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig_kalibrasi_detail_en.png", dpi=300, bbox_inches="tight"); plt.show()
