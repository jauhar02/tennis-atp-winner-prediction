#!/usr/bin/env python3
"""Compare two reproducibility manifests (a reference run and a new run).
Usage: python verify_against_reference.py reference/reproducibility_manifest.json new/reproducibility_manifest.json
Exit code 0 = data and results match; 1 = mismatch. Probabilities are compared after rounding to 6 decimals."""
import json, sys
ref, new = (json.load(open(p)) for p in sys.argv[1:3])
bad, warn = 0, 0
def chk(label, a, b, soft=False):
    global bad, warn
    ok = a == b
    print(f"[{'OK' if ok else ('WARN' if soft else 'FAIL')}] {label}" + ("" if ok else f"\n      reference: {a}\n      new:       {b}"))
    if not ok: warn += soft; bad += (not soft)
rf, nf = {f["file"]: f["sha256"] for f in ref["data"]["files"]}, {f["file"]: f["sha256"] for f in new["data"]["files"]}
chk("same set of data files", sorted(rf), sorted(nf))
chk("all data files byte-identical (sha256)", rf, nf)
for k in ("rows_raw", "rows_after_cleaning", "rows_after_cold_start", "hash_raw_rows", "hash_cleaned_rows", "hash_feature_matrix"):
    chk(f"data.{k}", ref["data"][k], new["data"][k])
for m in [k for k in ref["result_fingerprint"] if k != "n_test"]:
    a, b = ref["result_fingerprint"][m], new["result_fingerprint"][m]
    chk(f"{m}: correct predictions", a["correct_predictions"], b["correct_predictions"])
    chk(f"{m}: probabilities (rounded to 6 decimals)", a["sha256_probabilities_6dp"], b["sha256_probabilities_6dp"], soft=True)
_nt = lambda L: [{k: v for k, v in r.items() if k != "Waktu (s)"} for r in L]      # ignore wall-clock time
chk("selected hyperparameters per fold (CV AUC and parameters)", _nt(ref["hyperparameters_selected_per_fold"]), _nt(new["hyperparameters_selected_per_fold"]), soft=True)
chk("key package versions", ref["environment"]["key_packages"], new["environment"]["key_packages"], soft=True)
print(f"\n{'MATCH' if not bad else 'MISMATCH'}: {bad} failure(s), {warn} warning(s). Warnings usually mean different library versions or CPU floating-point behavior.")
sys.exit(1 if bad else 0)
