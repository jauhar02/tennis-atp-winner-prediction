#!/usr/bin/env python3
"""Run the full reproduction pipeline from the original match records.

Examples
  python run_all.py --data-dir ./data_snapshot --out ./outputs                 # archived snapshot (recommended)
  python run_all.py --raw-file ./data_snapshot/raw_records.pkl.gz --out ./outputs  # fallback snapshot format
  python run_all.py --data-revision <HF_COMMIT_SHA> --out ./outputs             # download a pinned Hugging Face revision
  python run_all.py --data-dir ./data_snapshot --out ./outputs --quick          # smoke test with tiny budgets (NOT the reported results)
  python run_all.py --data-dir ./data_snapshot --out ./outputs --from-step 5    # resume (needs earlier steps in the same run)

Steps (scripts/): 00 setup | 01 load+clean | 02 features | 03 split+info | 04 search-space calibration | 05 nested evaluation |
06 Table 4 core metrics | 07 extended tables (Table 4 baselines/ECE, Table 5) | 08 figures | 09 manifest.
"""
import argparse, os, sys, time, glob

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="", help="folder with atp_matches_YYYY.csv (archived snapshot)")
ap.add_argument("--raw-file", default="", help="pickled DataFrame with the exact raw records (fallback snapshot format)")
ap.add_argument("--data-revision", default="", help="Hugging Face dataset commit SHA to download")
ap.add_argument("--out", default="./outputs")
ap.add_argument("--quick", action="store_true", help="smoke test: tiny search budgets")
ap.add_argument("--fig-lang", default="en")
ap.add_argument("--only", default="", help="comma-separated step numbers, e.g. 0,1,2 (later steps need earlier ones in the same run)")
a = ap.parse_args()
if not a.data_dir and not a.data_revision and not a.raw_file:
    sys.exit("Give --data-dir (archived snapshot), --raw-file, or --data-revision (Hugging Face commit SHA). A floating 'main' download is not reproducible.")
os.environ.update(ATP_RAW_FILE=os.path.abspath(a.raw_file) if a.raw_file else "", ATP_DATA_DIR=os.path.abspath(a.data_dir) if a.data_dir else "", ATP_DATA_REVISION=a.data_revision,
                  ATP_OUT=os.path.abspath(a.out), ATP_QUICK="1" if a.quick else "0", ATP_FIG_LANG=a.fig_lang)
here = os.path.dirname(os.path.abspath(__file__)); sd = os.path.join(here, "scripts")
ns = {"__name__": "__repro__", "SCRIPTS_DIR": sd}
only = {int(x) for x in a.only.split(",") if x.strip()} if a.only else None
t0 = time.time()
for p in sorted(glob.glob(os.path.join(sd, "[0-9][0-9]_*.py"))):
    k = int(os.path.basename(p)[:2])
    if only is not None and k not in only: continue
    print(f"\n{'=' * 8} {os.path.basename(p)} ({(time.time() - t0) / 60:.1f} min elapsed) {'=' * 8}", flush=True)
    exec(compile(open(p).read(), p, "exec"), ns)
print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes. Outputs: {os.environ['ATP_OUT']}")
