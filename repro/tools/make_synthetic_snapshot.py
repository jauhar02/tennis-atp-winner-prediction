"""Write a small SYNTHETIC set of atp_matches_YYYY.csv files for smoke-testing the pipeline (python tools/make_synthetic_snapshot.py OUTDIR).
The data are random and carry no information about real players; they only exercise the code."""
import sys, os, numpy as np, pandas as pd
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
rg = np.random.default_rng(1); P = 70
ids = np.arange(1, P + 1); skill = rg.normal(0, 1, P); hand = np.where(rg.random(P) < .15, "L", "R"); ht = rg.integers(170, 205, P); age0 = rg.uniform(19, 33, P)
rows, tid = [], 0
for week in pd.date_range("2012-01-02", "2026-05-25", freq="7D"):
    tid += 1; rr = (tid % 9 == 0); surf = rg.choice(["Hard", "Clay", "Grass"]); lvl = "G" if tid % 13 == 0 else "A"; bo = 5 if lvl == "G" else 3
    pl = rg.permutation(P)[:32 if not rr else 8]; mn = [0]
    def play(a, b, rnd):
        mn[0] += 1; pa = 1 / (1 + np.exp(-(skill[a] - skill[b]))); w, l = (a, b) if rg.random() < pa else (b, a)
        rows.append(dict(tourney_id=f"{week.year}-{tid:04d}", tourney_date=int(week.strftime("%Y%m%d")), round=rnd, match_num=mn[0], winner_id=int(ids[w]), loser_id=int(ids[l]),
            winner_name=f"P{ids[w]}", loser_name=f"P{ids[l]}", winner_hand=hand[w], loser_hand=hand[l], winner_ht=float(ht[w]), loser_ht=float(ht[l]),
            winner_age=float(age0[w] + (week.year - 2012)), loser_age=float(age0[l] + (week.year - 2012)),
            winner_rank=int(150 - 20 * skill[w] + rg.integers(-5, 5)), loser_rank=int(150 - 20 * skill[l] + rg.integers(-5, 5)),
            winner_rank_points=float(1500 + 300 * skill[w]), loser_rank_points=float(1500 + 300 * skill[l]), surface=surf, best_of=bo, tourney_level=lvl,
            minutes=float(rg.integers(60, 200)), score="6-4 6-4"))
        return w
    if rr:
        for grp in (pl[:4], pl[4:]):
            for i in range(4):
                for j in range(i + 1, 4): play(grp[i], grp[j], "RR")
        play(pl[0], pl[5], "SF"); play(pl[1], pl[4], "SF"); play(pl[0], pl[1], "F")
    else:
        cur = list(pl)
        for rnd in ["R32", "R16", "QF", "SF", "F"]:
            cur = [play(cur[i], cur[i + 1], rnd) for i in range(0, len(cur), 2)]
df = pd.DataFrame(rows)
for col, p in (("winner_age", .08), ("loser_age", .08), ("winner_ht", .05), ("loser_ht", .05), ("minutes", .25), ("winner_rank_points", .01)):
    df.loc[rg.random(len(df)) < p, col] = np.nan
df["year"] = df.tourney_id.str[:4]
for y, part in df.groupby("year"): part.drop(columns="year").to_csv(os.path.join(out, f"atp_matches_{y}.csv"), index=False)
print("wrote", df.year.nunique(), "files,", len(df), "rows to", out)
