# ===== SEL AppCheck: BERAPA BANYAK LAGA UJI YANG AKAN DITOLAK ATAU DIBERI PERINGATAN OLEH PEMERIKSAAN APLIKASI (R2-14). Jalankan SETELAH sel aplikasi (butuh _cont, _lo, _hi, _ql, _qh, _mu, _P, MD_*, _combo, test, preds). =====
from IPython.display import display
_Xt = test[_cont].values.astype(float)
_out = ((_Xt < _lo) | (_Xt > _hi)).any(1)
_tail = ((_Xt < _ql) | (_Xt > _qh)).any(1) & ~_out
_mdv = np.sqrt(np.einsum("ij,jk,ik->i", _Xt - _mu, _P, _Xt - _mu))
_kc = np.array([_combo.get(tuple(int(v) for v in r), 0) for r in test[_BIN].values])
_blk = _out | (_mdv > MD_BLOCK) | (_kc == 0)
_cau = ~_blk & (_tail | (_mdv > MD_CAUTION) | (_kc < MIN_COMBO))
_st = np.where(_blk, "Refused", np.where(_cau, "Caution", "Supported"))
_yv = test.target.values; _yrs = test.date.dt.year.values
print(f"Batas pada data latih ({len(_trn):,} laga, {TRAIN_START_YEAR}-2022): jarak Mahalanobis 99% = {MD_CAUTION:.2f}, 99.9% = {MD_BLOCK:.2f}")
print(f"Rentang Elo pemain = {ELO_R[0]:.0f} sampai {ELO_R[1]:.0f}; usia {AGE_R[0]:.1f} sampai {AGE_R[1]:.1f}; tinggi {HT_R[0]:.0f} sampai {HT_R[1]:.0f}; ranking maks {RANK_MAX:.0f}")
_rows = []
for lab in ("Supported", "Caution", "Refused"):
    m = _st == lab
    _rows.append({"Status": lab, "Matches": int(m.sum()), "Share (%)": round(100 * m.mean(), 2),
                  **{f"Accuracy {n} (%)": (round(100 * float((preds[n][m] == _yv[m]).mean()), 2) if m.sum() else np.nan) for n in preds}})
t_appchk = pd.DataFrame(_rows); display(t_appchk)
_by = pd.crosstab(_yrs, _st); display(_by)
_why = pd.Series({"Feature outside the range seen in training": int(_out.sum()), "Mahalanobis distance above the 99.9% limit": int((_mdv > MD_BLOCK).sum()),
                  "Format-level-surface combination absent in training": int((_kc == 0).sum()), "Caution only: extreme tail or Mahalanobis above 99%": int(_cau.sum())})
display(_why.to_frame("Test matches"))
t_appchk.to_csv(f"{OUT_DIR}/tabel_aplikasi_pemeriksaan_laga_uji.csv", index=False)
print("Tersimpan: tabel_aplikasi_pemeriksaan_laga_uji.csv")
