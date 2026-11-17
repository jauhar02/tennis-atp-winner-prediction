# Part of the reproduction pipeline for 'A Comparison of Random Forest, XGBoost, and SVM in Predicting ATP Tennis Match Outcomes'.
# Scripts are executed in order by run_all.py in one shared namespace (they are also notebook cells: paste them in order).
def _elo0(): return 1500.0
def _deque10(): return deque(maxlen=10)
def _deque60(): return deque(maxlen=60)
def _list(): return []

def expected(a, b):
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))

class Tracker:
    """Menyimpan status setiap pemain (Elo, form, H2H, kelelahan, dll.) secara kronologis."""
    def __init__(self):
        self.elo = defaultdict(_elo0)          # Elo umum
        self.elo_s = defaultdict(_elo0)        # Elo per permukaan: key (pid, surface)
        self.n = defaultdict(int)              # jumlah laga
        self.n_s = defaultdict(int)            # jumlah laga per permukaan
        self.wins_s = defaultdict(int)         # kemenangan per permukaan
        self.form = defaultdict(_deque10)      # hasil 10 laga terakhir (1/0)
        self.load = defaultdict(_deque60)      # (tanggal, menit bermain)
        self.h2h = defaultdict(int)            # (a, b) -> jumlah kemenangan a atas b
        self.bio = {}                          # data terakhir pemain (untuk prediksi)
        self.elo_hist = defaultdict(_list)     # riwayat Elo untuk grafik

    def dynamic(self, pid, surface, date):
        """Fitur dinamis pemain SEBELUM pertandingan pada tanggal `date`."""
        ns = self.n_s[(pid, surface)]
        fatigue = sum(m for d, m in self.load[pid] if 0 <= (date - d).days <= 14)
        return dict(
            elo=self.elo[pid], elo_s=self.elo_s[(pid, surface)],
            form=float(np.mean(self.form[pid])) if self.form[pid] else 0.5,
            surf_wr=(self.wins_s[(pid, surface)] + 2.5) / (ns + 5),   # dihaluskan (prior 50%)
            fatigue=float(fatigue), exp=self.n[pid])

    def update(self, r):
        """Perbarui status SETELAH pertandingan selesai."""
        w, l, s, d = r.winner_id, r.loser_id, r.surface, r.tourney_date
        mins = r.minutes if pd.notna(r.minutes) else 100
        # Elo umum dan per permukaan (K menurun seiring jumlah laga; nilai ditetapkan penulis)
        for store, kw_n, kl_n, kw, kl in (
            (self.elo,   self.n[w],        self.n[l],        w,      l),
            (self.elo_s, self.n_s[(w, s)], self.n_s[(l, s)], (w, s), (l, s))):
            e_w = expected(store[kw], store[kl])
            store[kw] += 250 / ((kw_n + 5) ** 0.4) * (1 - e_w)
            store[kl] -= 250 / ((kl_n + 5) ** 0.4) * (1 - e_w)
        for pid, won in ((w, 1), (l, 0)):
            self.n[pid] += 1; self.n_s[(pid, s)] += 1; self.wins_s[(pid, s)] += won
            self.form[pid].append(won); self.load[pid].append((d, mins))
            self.elo_hist[pid].append((d, self.elo[pid]))
        self.h2h[(w, l)] += 1
        for pref, pid in (("winner", w), ("loser", l)):
            self.bio[pid] = dict(name=getattr(r, f"{pref}_name"), hand=getattr(r, f"{pref}_hand"),
                                 ht=getattr(r, f"{pref}_ht"), age=getattr(r, f"{pref}_age"),
                                 rank=getattr(r, f"{pref}_rank"), points=getattr(r, f"{pref}_rank_points"),
                                 last_date=d)

FEATURES = ["elo_diff", "elo_surface_diff", "rank_diff_log", "points_diff_log", "age_diff", "height_diff",
            "form_diff", "surface_wr_diff", "h2h_diff", "fatigue_diff", "experience_diff", "lefty_diff",
            "best_of_5", "is_slam", "surf_hard", "surf_clay", "surf_grass"]

def _diff(a, b):
    return 0.0 if (pd.isna(a) or pd.isna(b)) else float(a - b)

def build_features(p1, p2, surface, best_of, is_slam, h2h_diff):
    """Satu-satunya fungsi pembentuk fitur, dipakai saat pelatihan DAN saat prediksi (anti train/serve skew).
    Nilai positif = menguntungkan Pemain 1."""
    r1, r2 = p1.get("rank", np.nan), p2.get("rank", np.nan)
    rank_ok = pd.notna(r1) and pd.notna(r2) and r1 > 0 and r2 > 0
    pt1, pt2 = p1.get("points", np.nan), p2.get("points", np.nan)
    return dict(
        elo_diff=p1["elo"] - p2["elo"],
        elo_surface_diff=p1["elo_s"] - p2["elo_s"],
        rank_diff_log=float(np.log(r2 / r1)) if rank_ok else 0.0,
        points_diff_log=_diff(np.log1p(pt1), np.log1p(pt2)) if pd.notna(pt1) and pd.notna(pt2) else 0.0,
        age_diff=_diff(p1.get("age", np.nan), p2.get("age", np.nan)),
        height_diff=_diff(p1.get("ht", np.nan), p2.get("ht", np.nan)),
        form_diff=p1["form"] - p2["form"],
        surface_wr_diff=p1["surf_wr"] - p2["surf_wr"],
        h2h_diff=float(h2h_diff),
        fatigue_diff=(p1["fatigue"] - p2["fatigue"]) / 60.0,        # dalam jam
        experience_diff=float(np.log1p(p1["exp"]) - np.log1p(p2["exp"])),
        lefty_diff=p1.get("lefty", 0.0) - p2.get("lefty", 0.0),
        best_of_5=int(best_of == 5), is_slam=int(is_slam),
        surf_hard=int(surface == "Hard"), surf_clay=int(surface == "Clay"), surf_grass=int(surface == "Grass"))

tr = Tracker()
rng = np.random.default_rng(SEED)
rows = []

for r in df.itertuples(index=False):
    s, d = r.surface, r.tourney_date
    w = dict(rank=r.winner_rank, points=r.winner_rank_points, age=r.winner_age, ht=r.winner_ht,
             lefty=float(r.winner_hand == "L"), **tr.dynamic(r.winner_id, s, d))
    l = dict(rank=r.loser_rank, points=r.loser_rank_points, age=r.loser_age, ht=r.loser_ht,
             lefty=float(r.loser_hand == "L"), **tr.dynamic(r.loser_id, s, d))
    h2h_w = tr.h2h[(r.winner_id, r.loser_id)] - tr.h2h[(r.loser_id, r.winner_id)]

    # acak posisi: p1 = pemenang (target 1) atau p1 = kalah (target 0)
    if rng.random() < 0.5:
        p1, p2, h2h, y = w, l, h2h_w, 1
    else:
        p1, p2, h2h, y = l, w, -h2h_w, 0

    feats = build_features(p1, p2, s, r.best_of, r.tourney_level == "G", h2h)
    feats.update(target=y, date=d, n1=p1["exp"], n2=p2["exp"])
    rows.append(feats)
    tr.update(r)                       # update SETELAH fitur dihitung -> tidak ada leakage

data = pd.DataFrame(rows)
data = data[(data.n1 >= MIN_PRIOR_MATCHES) & (data.n2 >= MIN_PRIOR_MATCHES)].reset_index(drop=True)
print("Dataset akhir:", data.shape, "| proporsi target=1:", round(data.target.mean(), 3))
data.head()
