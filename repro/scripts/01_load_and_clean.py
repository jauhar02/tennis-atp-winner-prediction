# Part of the reproduction pipeline for 'A Comparison of Random Forest, XGBoost, and SVM in Predicting ATP Tennis Match Outcomes'.
# Scripts are executed in order by run_all.py in one shared namespace (they are also notebook cells: paste them in order).
file_hashes, frames = [], []
if RAW_FILE:
    with open(RAW_FILE, "rb") as fh: _b = fh.read()
    frames.append(pd.read_pickle(RAW_FILE))
    file_hashes.append({"file": os.path.basename(RAW_FILE), "bytes": len(_b), "sha256": hashlib.sha256(_b).hexdigest(), "rows": int(len(frames[0]))})
for y in ([] if RAW_FILE else range(START_YEAR, END_YEAR + 1)):
    name = f"atp_matches_{y}.csv"
    try:
        if DATA_DIR:
            with open(os.path.join(DATA_DIR, name), "rb") as fh: b = fh.read()
        else:
            import requests
            resp = requests.get(f"{DATA_URL}/{name}", timeout=120); resp.raise_for_status(); b = resp.content
        frames.append(pd.read_csv(io.BytesIO(b)))
        file_hashes.append({"file": name, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "rows": int(len(frames[-1]))})
    except Exception as e:
        print(f"Failed to load {y}: {type(e).__name__}: {str(e)[:100]}")

if not frames:
    raise RuntimeError("No data could be loaded. Check ATP_DATA_DIR or the internet connection, then rerun.")
if not DATA_DIR and not DATA_REVISION and not RAW_FILE:
    print("WARNING: ATP_DATA_REVISION is empty, so the files come from the floating 'main' branch and may differ from the reported snapshot.")
print("Files loaded:", len(frames))
raw = pd.concat(frames, ignore_index=True)
print("Raw rows:", len(raw))

df = raw.copy()
df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(int).astype(str), format="%Y%m%d")

# buang baris tanpa info penting
df = df.dropna(subset=["surface", "winner_id", "loser_id", "winner_rank", "loser_rank"])

# buang walkover / default (dan opsional retired)
score = df["score"].fillna("")
bad = score.str.contains("W/O|DEF")
if DROP_RETIRED:
    bad |= score.str.contains("RET")
df = df[~bad]

# urutkan kronologis (penting untuk feature engineering tanpa leakage)
ROUND_ORDER = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "RR": 4, "QF": 5, "SF": 6, "BR": 6, "F": 7}
df["round_order"] = df["round"].map(ROUND_ORDER).fillna(0)
df = df.sort_values(["tourney_date", "round_order", "match_num"]).reset_index(drop=True)
print("Baris setelah pembersihan:", len(df))
df[["tourney_date", "surface", "round", "winner_name", "loser_name", "winner_rank", "loser_rank"]].head()
