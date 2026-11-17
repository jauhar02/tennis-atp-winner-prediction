# ===== SEL APLIKASI GRADIO DENGAN PEMERIKSAAN INPUT (R2-14). Pengganti sel Gradio lama. =====
# Butuh: df, data, tr, final_models, FEATURES, build_features, MIN_PRIOR_MATCHES, TRAIN_START_YEAR, TEST_START_DATE (semuanya ada setelah restore checkpoint dan sel Tracker).
# Tiga hal baru: (1) tanggal acuan eksplisit, (2) pemeriksaan pemain dan input (riwayat, usia/ranking basi, kustom tanpa riwayat), (3) deteksi di luar sebaran data latih.
# Bila input tidak didukung, probabilitas TIDAK ditampilkan; bila meragukan, probabilitas ditampilkan dengan peringatan.
import gradio as gr
from sklearn.metrics import log_loss

MIN_MATCHES = MIN_PRIOR_MATCHES          # sama dengan aturan cold-start pada pelatihan
STALE_DAYS = 180                         # laga terakhir pemain lebih lama dari ini: ranking dan usia mungkin usang (peringatan)
MIN_SURF_MATCHES = 5                     # laga di permukaan ini kurang dari ini: Elo/win rate permukaan lemah (peringatan)
MIN_COMBO = 30                           # kombinasi format-level-permukaan yang muncul kurang dari ini pada latih (peringatan; 0 = ditolak)
DATE_CAUTION_DAYS, DATE_BLOCK_DAYS = 30, 180      # tanggal laga setelah batas data: peringatan, lalu ditolak
Q_TAIL = (0.005, 0.995)                  # ekor sebaran latih per fitur (peringatan)
LEVELS = {"Grand Slam": True, "Other tournaments": False}
SURFACES = ["Hard", "Clay", "Grass"]

REF_DATE = df.tourney_date.max()                         # tanggal acuan status pemain = tanggal turnamen terakhir pada data
_trn = data[(data.date >= f"{TRAIN_START_YEAR}-01-01") & (data.date < TEST_START_DATE)]      # data latih final_models (lipatan 1)
try: BEST_MODEL = results.loc[list(final_models)].LogLoss.idxmin()
except Exception: BEST_MODEL = min(final_models, key=lambda m: log_loss(y_test, probas[m]))

# ---------- referensi sebaran data latih untuk deteksi di luar sebaran ----------
_BIN = ["best_of_5", "is_slam", "surf_hard", "surf_clay", "surf_grass"]
_cont = [f for f in FEATURES if f not in _BIN]
_Xc = _trn[_cont].values.astype(float); _mu = _Xc.mean(0); _P = np.linalg.pinv(np.cov(_Xc, rowvar=False))
_lo, _hi = _Xc.min(0), _Xc.max(0); _ql, _qh = np.quantile(_Xc, Q_TAIL[0], axis=0), np.quantile(_Xc, Q_TAIL[1], axis=0)
_mdt = np.sqrt(np.einsum("ij,jk,ik->i", _Xc - _mu, _P, _Xc - _mu)); MD_CAUTION, MD_BLOCK = np.quantile(_mdt, [0.99, 0.999])
_combo = _trn.groupby(_BIN).size().to_dict()
_d_ = df[df.tourney_date >= f"{TRAIN_START_YEAR}-01-01"]
AGE_R = tuple(pd.concat([_d_.winner_age, _d_.loser_age]).dropna().quantile([0.001, 0.999]))
HT_R = tuple(pd.concat([_d_.winner_ht, _d_.loser_ht]).dropna().quantile([0.001, 0.999]))
RANK_MAX = float(pd.concat([_d_.winner_rank, _d_.loser_rank]).dropna().max())
_el = [tr.elo[p] for p in tr.n if tr.n[p] >= MIN_MATCHES]; ELO_R = (min(_el), max(_el))

# ---------- daftar pemain terdaftar: >= MIN_MATCHES laga dan bermain dalam 2 tahun terakhir data ----------
active = {pid: b["name"] for pid, b in tr.bio.items() if tr.n[pid] >= MIN_MATCHES and (REF_DATE - b["last_date"]).days <= 730}
name_to_id = {n: pid for pid, n in sorted(active.items(), key=lambda x: -tr.elo[x[0]])}
player_names = list(name_to_id)

def profile_registered(name, surface):
    """Status pemain pada tanggal acuan: Elo, form, win rate, kelelahan 14 hari dan jumlah laga dari Tracker;
    ranking, poin, tinggi, tangan dari laga terakhir; usia = usia pada laga terakhir ditambah selang waktu sampai tanggal acuan."""
    pid = name_to_id[name]; b = tr.bio[pid]; gap = (REF_DATE - b["last_date"]).days
    p = dict(rank=b["rank"], points=b["points"], age=(b["age"] + gap / 365.25) if pd.notna(b["age"]) else np.nan,
             ht=b["ht"], lefty=float(b["hand"] == "L"))
    p.update(tr.dynamic(pid, surface, REF_DATE))
    return pid, p, dict(n=tr.n[pid], last=b["last_date"], gap=gap, surf_n=tr.n_s[(pid, surface)])

def profile_custom(elo, rank, age, ht, hand, form_pct, surf_wr_pct, fatigue_h, n_matches):
    return dict(rank=rank if rank and rank > 0 else np.nan, points=np.nan, age=age, ht=ht, lefty=float(hand == "L"),
                elo=elo, elo_s=elo, form=(form_pct or 0) / 100, surf_wr=(surf_wr_pct or 0) / 100,
                fatigue=(fatigue_h or 0) * 60, exp=int(n_matches or 0))

def predict_pair(p1, p2, surface, best_of, is_slam, h2h_diff, model_name):
    """Rata-rata dua urutan (A,B) dan (B,A) agar hasil tidak bergantung pada urutan pemain."""
    m = final_models[model_name]
    Xa = pd.DataFrame([build_features(p1, p2, surface, best_of, is_slam, h2h_diff)])[FEATURES]
    Xb = pd.DataFrame([build_features(p2, p1, surface, best_of, is_slam, -h2h_diff)])[FEATURES]
    return float((m.predict_proba(Xa)[0, 1] + 1 - m.predict_proba(Xb)[0, 1]) / 2)

# ---------- pemeriksaan ----------
def check_registered(name, m, surface, p):
    B, C = [], []
    if m["n"] < MIN_MATCHES: B.append(f"{name}: only {m['n']} recorded matches (minimum {MIN_MATCHES}); the models were trained only on players with at least {MIN_MATCHES} earlier matches.")
    if m["gap"] > STALE_DAYS: C.append(f"{name}: last recorded match on {m['last'].date()}, {m['gap']} days before the reference date; ranking and height come from that match and may be outdated, and age was extrapolated.")
    if pd.isna(p["rank"]): C.append(f"{name}: no ranking on record; the ranking difference is set to 0, as in training.")
    if m["surf_n"] < MIN_SURF_MATCHES: C.append(f"{name}: only {m['surf_n']} recorded matches on {surface}; the surface Elo and win rate rest on little data.")
    return B, C

def check_custom(name, p):
    B, C = [], ["{}: profile entered by the user; it is not verified against match records.".format(name)]
    if not p["exp"] or p["exp"] < MIN_MATCHES or pd.isna(p["elo"]): B.append(f"{name}: no validated match history (fewer than {MIN_MATCHES} matches recorded, or Elo missing); the models were trained only on players with at least {MIN_MATCHES} earlier matches.")
    for lab, v, (a, b_) in (("Elo", p["elo"], ELO_R), ("Age", p["age"], AGE_R), ("Height", p["ht"], HT_R)):
        if pd.isna(v): B.append(f"{name}: {lab} is missing.")
        elif not (a <= v <= b_): B.append(f"{name}: {lab} {v:.0f} is outside the range of players in the data ({a:.0f} to {b_:.0f}).")
    if pd.notna(p["rank"]) and p["rank"] > RANK_MAX: B.append(f"{name}: ranking {p['rank']:.0f} is worse than any ranking in the data (maximum {RANK_MAX:.0f}).")
    return B, C

def check_features(f, surface, best_of, is_slam):
    B, C = [], []
    x = np.array([f[c] for c in _cont], float)
    out = [c for c, v, a, b_ in zip(_cont, x, _lo, _hi) if not (a <= v <= b_)]
    tail = [c for c, v, a, b_ in zip(_cont, x, _ql, _qh) if not (a <= v <= b_) and c not in out]
    if out: B.append("Outside the range seen in training: " + ", ".join(out) + ".")
    if tail: C.append("In the extreme tail of the training distribution (outside its 0.5% to 99.5% range): " + ", ".join(tail) + ".")
    d = float(np.sqrt((x - _mu) @ _P @ (x - _mu)))
    if d > MD_BLOCK: B.append(f"The combination of the 12 continuous features is more unusual than 99.9% of training matches (Mahalanobis distance {d:.1f}).")
    elif d > MD_CAUTION: C.append(f"The combination of the 12 continuous features is more unusual than 99% of training matches (Mahalanobis distance {d:.1f}).")
    k = _combo.get((int(best_of == 5), int(is_slam), int(surface == "Hard"), int(surface == "Clay"), int(surface == "Grass")), 0)
    if k == 0: B.append("This combination of match format, tournament level, and surface never occurred in training (Grand Slam men's matches are best-of-5).")
    elif k < MIN_COMBO: C.append(f"This combination of format, level, and surface occurred in only {k} training matches.")
    return B, C

def check_date(txt):
    if txt is None or str(txt).strip() == "": return [], []
    try: dt = pd.Timestamp(str(txt).strip())
    except Exception: return [f"Match date '{txt}' is not a valid date (use YYYY-MM-DD)."], []
    k = (dt - REF_DATE).days
    if k < 0: return [f"The state of the players is available only at the data cut-off ({REF_DATE.date()}); earlier dates cannot be reproduced here."], []
    if k > DATE_BLOCK_DAYS: return [f"The match date is {k} days after the data cut-off ({REF_DATE.date()}); Elo, form, and fatigue would be badly out of date."], []
    if k > DATE_CAUTION_DAYS: return [], [f"The match date is {k} days after the data cut-off ({REF_DATE.date()}); matches played since then are not in the Elo, form, and fatigue."]
    return [], []

def feature_table(f):
    rows = []
    for c in FEATURES:
        v = f[c]
        if c in _cont:
            i = _cont.index(c); st = "yes" if _ql[i] <= v <= _qh[i] else ("OUTSIDE" if not (_lo[i] <= v <= _hi[i]) else "tail")
            rows.append([c, round(float(v), 3), f"{_ql[i]:.2f} to {_qh[i]:.2f}", st])
        else: rows.append([c, int(v), "0 or 1", "yes"])
    return pd.DataFrame(rows, columns=["Feature (17)", "Value", "Training range (0.5% to 99.5%)", "Within range"])

def compare_table(n1, p1, n2, p2):
    def _r(v, k=0): return "-" if v is None or pd.isna(v) else (round(v, k) if k else int(round(v)))
    def fmt(p): return [_r(p["elo"]), _r(p["elo_s"]), _r(p.get("rank")), f"{p['form']:.0%}",
                        f"{p['surf_wr']:.0%}", round(p["fatigue"] / 60, 1), int(p["exp"])]
    return pd.DataFrame({"Factor": ["Elo (general)", "Elo (surface)", "Ranking", "Form (last 10)", "Surface win rate", "Hours played (14 days)", "Matches"],
                         n1: fmt(p1), n2: fmt(p2)})

def plot_prob(n1, n2, prob):
    fig, ax = plt.subplots(figsize=(7, 1.9))
    ax.barh(0, prob, color="#2a9d8f"); ax.barh(0, 1 - prob, left=prob, color="#e76f51")
    ax.text(prob / 2, 0, f"{n1}\n{prob:.1%}", ha="center", va="center", color="white", fontweight="bold")
    ax.text(prob + (1 - prob) / 2, 0, f"{n2}\n{1 - prob:.1%}", ha="center", va="center", color="white", fontweight="bold")
    ax.axvline(0.5, color="white", ls="--", lw=1); ax.set_xlim(0, 1); ax.axis("off")
    ax.set_title("Win probability", fontsize=11); plt.tight_layout(); return fig

def plot_elo(id1, n1, id2, n2):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for pid, n in ((id1, n1), (id2, n2)):
        h = tr.elo_hist[pid]
        if h: ax.plot([d for d, _ in h], [e for _, e in h], label=n)
    ax.set(title="Elo trend", ylabel="Elo"); ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); return fig

def _report(n1, n2, prob, B, C, head, foot):
    if B: return ("### No probability shown: unsupported input\n" + "\n".join(f"- {m}" for m in B) +
                  ("\n\nAlso noted:\n" + "\n".join(f"- {m}" for m in C) if C else "") +
                  "\n\n*The models were trained on ATP tour matches of players with at least 10 earlier matches; outside that population an estimate would be an extrapolation.*")
    s = f"### {n1}: **{prob:.1%}**  vs  {n2}: **{1 - prob:.1%}**\n{head}\n\n"
    s += ("**Use with caution:**\n" + "\n".join(f"- {m}" for m in C) + "\n\n") if C else "**Status:** all checks passed.\n\n"
    return s + foot

_STATE = f"Player states are those after the last match in the data ({REF_DATE.date()}). Models trained on {len(_trn):,} matches from {_trn.date.min().date()} to {_trn.date.max().date()}."

def run_registered(n1, n2, surface, level, best_of, model_name, date_txt):
    if not n1 or not n2 or n1 == n2: return None, None, "Pick two different players.", None, None
    id1, p1, m1 = profile_registered(n1, surface); id2, p2, m2 = profile_registered(n2, surface)
    h2h = tr.h2h[(id1, id2)] - tr.h2h[(id2, id1)]
    f = build_features(p1, p2, surface, int(best_of), LEVELS[level], h2h)
    B, C = [], []
    for nm, m, p in ((n1, m1, p1), (n2, m2, p2)): b, c = check_registered(nm, m, surface, p); B += b; C += c
    for chk in (check_features(f, surface, int(best_of), LEVELS[level]), check_date(date_txt)): B += chk[0]; C += chk[1]
    prob = None if B else predict_pair(p1, p2, surface, int(best_of), LEVELS[level], h2h, model_name)
    head = f"Model: {model_name} | Surface: {surface} | Head-to-head (win difference): {h2h:+d}"
    foot = f"*{_STATE} For academic use, not for betting.*"
    return (None if B else plot_prob(n1, n2, prob)), plot_elo(id1, n1, id2, n2), _report(n1, n2, prob, B, C, head, foot), compare_table(n1, p1, n2, p2), feature_table(f)

def run_custom(n1, e1, r1, a1, h1, hd1, f1, s1, fat1, m1, n2, e2, r2, a2, h2, hd2, f2, s2, fat2, m2, surface, level, best_of, model_name, date_txt):
    n1, n2 = n1 or "Player A", n2 or "Player B"
    p1 = profile_custom(e1, r1, a1, h1, hd1, f1, s1, fat1, m1); p2 = profile_custom(e2, r2, a2, h2, hd2, f2, s2, fat2, m2)
    B, C = [], []
    for nm, p in ((n1, p1), (n2, p2)): b, c = check_custom(nm, p); B += b; C += c
    f = None
    if not any(pd.isna(p[k]) for p in (p1, p2) for k in ("elo", "age", "ht")):
        f = build_features(p1, p2, surface, int(best_of), LEVELS[level], 0)
        b, c = check_features(f, surface, int(best_of), LEVELS[level]); B += b; C += c
    b, c = check_date(date_txt); B += b; C += c
    prob = None if B else predict_pair(p1, p2, surface, int(best_of), LEVELS[level], 0, model_name)
    head = f"Model: {model_name} | Surface: {surface} | Head-to-head set to 0 and ranking points missing (as in training when unknown)."
    foot = "*Estimate for a user-entered profile; not validated for amateur players. " + _STATE + "*"
    return (None if B else plot_prob(n1, n2, prob)), _report(n1, n2, prob, B, C, head, foot), compare_table(n1, p1, n2, p2), (feature_table(f) if f else None)

def player_inputs(tag, default_name):
    with gr.Column():
        gr.Markdown(f"**{tag}**")
        return [gr.Textbox(label="Name", value=default_name),
                gr.Number(label="Internal Elo (new player = 1500)", value=1500),
                gr.Number(label="Ranking (leave empty if none)", value=None),
                gr.Number(label="Age", value=25), gr.Number(label="Height (cm)", value=175),
                gr.Radio(["R", "L"], label="Dominant hand", value="R"),
                gr.Slider(0, 100, value=50, label="Win % in last 10 matches"),
                gr.Slider(0, 100, value=50, label="Win % on this surface"),
                gr.Number(label="Hours played in last 14 days", value=0),
                gr.Number(label="Recorded matches (at least 10)", value=15)]

with gr.Blocks(title="Tennis Match Winner Prediction") as demo:
    gr.Markdown("# Tennis Match Winner Prediction\nRandom Forest, XGBoost, and SVM | ATP data (Jeff Sackmann)\n\n" + _STATE)
    with gr.Row():
        surface = gr.Radio(SURFACES, label="Surface", value="Hard")
        level = gr.Radio(list(LEVELS), label="Tournament level", value="Other tournaments")
        best_of = gr.Radio([3, 5], label="Best of", value=3)
        model_name = gr.Dropdown(list(final_models), label="Model", value=BEST_MODEL)
        date_in = gr.Textbox(label="Match date (YYYY-MM-DD)", value=str(REF_DATE.date()))
    with gr.Tab("Registered players"):
        with gr.Row():
            d1 = gr.Dropdown(player_names, label="Player A", value=player_names[0])
            d2 = gr.Dropdown(player_names, label="Player B", value=player_names[1])
        b1 = gr.Button("Predict", variant="primary")
        t1 = gr.Markdown(); pl1 = gr.Plot(); pl2 = gr.Plot(); tb1 = gr.Dataframe(label="Factor comparison"); tf1 = gr.Dataframe(label="The 17 features given to the model")
        b1.click(run_registered, [d1, d2, surface, level, best_of, model_name, date_in], [pl1, pl2, t1, tb1, tf1])
    with gr.Tab("Custom players"):
        with gr.Row():
            in1 = player_inputs("Player A", "Player A"); in2 = player_inputs("Player B", "Player B")
        b2 = gr.Button("Predict", variant="primary")
        t2 = gr.Markdown(); pl3 = gr.Plot(); tb2 = gr.Dataframe(label="Factor comparison"); tf2 = gr.Dataframe(label="The 17 features given to the model")
        b2.click(run_custom, in1 + in2 + [surface, level, best_of, model_name, date_in], [pl3, t2, tb2, tf2])

demo.launch(share=True)
