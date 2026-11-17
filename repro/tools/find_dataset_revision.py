"""Find which Hugging Face revision of the dataset matches the match records used in the experiments, and export an exact snapshot.
Run inside the session where `raw` (the concatenated raw match records used in the experiments) is in memory, e.g. after restoring the checkpoint:

    exec(open("tools/find_dataset_revision.py").read())
    sha = find_revision(raw)                         # searches the recent commit history; prints the matching commit SHA, if any
    export_snapshot(raw, "data_snapshot")            # always do this: writes atp_matches_YYYY.csv files that reproduce `raw` exactly

Archive the data_snapshot folder (for example on Zenodo; the data are CC BY-NC-SA 4.0) and cite the DOI, together with the commit SHA when found.
"""
import io, os, requests, numpy as np, pandas as pd
_REPO = "Aneeshers/tennis-sackmann-archive"
def _hash(d): return int(pd.util.hash_pandas_object(d, index=False).values.sum(dtype=np.uint64))

def find_revision(raw, repo=_REPO, max_commits=40, years=range(2000, 2027)):
    target, n = _hash(raw), len(raw)
    commits = requests.get(f"https://huggingface.co/api/datasets/{repo}/commits/main", timeout=30).json()
    print(f"Checking {min(len(commits), max_commits)} commits for a snapshot with {n:,} rows and hash {target} ...")
    for c in commits[:max_commits]:
        sha, frames = c["id"], []
        for y in years:
            r = requests.get(f"https://huggingface.co/datasets/{repo}/resolve/{sha}/atp/atp_matches_{y}.csv", timeout=120)
            if r.status_code == 200: frames.append(pd.read_csv(io.BytesIO(r.content)))
        cur = pd.concat(frames, ignore_index=True) if frames else None
        same = cur is not None and len(cur) == n and _hash(cur) == target
        print(f"  {sha[:10]}  {str(c.get('date', ''))[:10]}  rows={0 if cur is None else len(cur):,}  {'MATCH' if same else 'no'}")
        if same: return sha
    print("No match in the searched history. Use export_snapshot(raw, ...) and archive that snapshot instead.")
    return None

def export_snapshot(raw, outdir, year_col="tourney_id"):
    """Write one CSV per season (season = first four characters of tourney_id), then read them back and check that they reproduce `raw`."""
    os.makedirs(outdir, exist_ok=True)
    yrs = raw[year_col].astype(str).str[:4]
    for y, part in raw.groupby(yrs, sort=True): part.to_csv(os.path.join(outdir, f"atp_matches_{y}.csv"), index=False)
    back = pd.concat([pd.read_csv(os.path.join(outdir, f"atp_matches_{y}.csv")) for y in sorted(yrs.unique())], ignore_index=True)
    ok = len(back) == len(raw) and _hash(back) == _hash(raw)
    print(f"Snapshot written to {outdir}: {yrs.nunique()} files, {len(back):,} rows; reproduces raw exactly: {ok}")
    if not ok:
        p = os.path.join(outdir, "raw_records.pkl.gz"); raw.to_pickle(p)
        print(f"CSV text does not reproduce raw bit for bit (pandas' default float parser). Wrote {p} instead; archive this file and run: python run_all.py --raw-file {p} ...")
        print("A pickle depends on the pandas version; record it (the manifest does) or also archive a Parquet copy: raw.to_parquet('raw_records.parquet').")
    return ok
