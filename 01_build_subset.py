#!/usr/bin/env python
"""Cycle 1: load TriviaQA validation, deduplicate, validate, sample the pool, split, save ids only.

Run from the repo root:  python scripts/01_build_subset.py
Options: --pool-size N   (ONLY for the pre-declared 400 fallback; the reason must be logged in protocol_changes.md)
         --no-write      (print everything but do not write files)
Outputs: data/subset_question_ids.csv, data/subset_manifest.json, logs/01_build_subset_output.txt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uncertainty import data as D  # noqa: E402


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for f in self.s:
            f.write(x)
            f.flush()

    def flush(self):
        for f in self.s:
            f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-size", type=int, default=None)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
    log = open(ROOT / "logs" / "01_build_subset_output.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log)

    cfg = yaml.safe_load(open(ROOT / "configs" / "protocol_v1.yaml"))
    ds_cfg = cfg["dataset"]
    print("CYCLE 1 | build subset |", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("protocol:", cfg["protocol_version"], "| status:", cfg["status"])
    if args.pool_size:
        print(f"WARNING: pool size overridden to {args.pool_size} (allowed only for the declared fallback; log it).")

    from datasets import load_dataset
    ds = load_dataset(ds_cfg["hf_name"], ds_cfg["config"], split=ds_cfg["split"])
    print(f"loaded {ds_cfg['hf_name']} / {ds_cfg['config']} / {ds_cfg['split']}: {len(ds)} rows")
    df = D.rows_from_records(ds)
    print("columns:", list(df.columns))
    print("question_source values:", df["question_source"].value_counts().head(5).to_dict())
    print("problems BEFORE cleaning:", D.validate(df))

    sub, rep = D.build_subset(df, ds_cfg, args.pool_size)
    print("\n--- cleaning ---")
    for k, v in rep["dedupe"].items():
        print(f"{k:<42}{v}")
    print("problems AFTER dedupe:", rep["validation_after_dedupe"])
    print("rows dropped for having no alias:", rep["dropped_unlabellable"])
    print("questions available after cleaning:", rep["available_after_cleaning"])

    print("\n--- subset ---")
    print(f"pool size {rep['pool_size']} | calibration {rep['n_calibration']} | test {rep['n_test']} | pilot {rep['n_pilot']}")
    assert rep["n_calibration"] + rep["n_test"] == rep["pool_size"]
    pilot_splits = set(sub.loc[sub["is_pilot"], "split"])
    print("pilot rows belong to split(s):", pilot_splits)
    assert pilot_splits <= {"calibration"}, "pilot must come from calibration only"
    ids = D.ids_table(sub)
    assert ids["question_id"].is_unique
    print("first 3 pool rows:")
    for r in sub.head(3).itertuples():
        print(f"  #{r.pool_index} id={r.question_id} split={r.split} pilot={r.is_pilot} | {r.question[:70]!r} -> {r.gold_value!r}")
    print("alias count per question (min/median/max):",
          int(sub['normalized_aliases'].map(len).min()), int(sub['normalized_aliases'].map(len).median()), int(sub['normalized_aliases'].map(len).max()))

    # determinism self-check: shuffle the input rows and rebuild; the id table must be identical
    shuffled = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    sub2, rep2 = D.build_subset(shuffled, ds_cfg, args.pool_size)
    same = rep2["ids_sha256"] == rep["ids_sha256"]
    print("\ndeterminism self-check (shuffled input gives the same ids):", same)
    print("ids sha256:", rep["ids_sha256"])
    assert same, "subset depends on row order: bug"

    if not args.no_write:
        extra = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "protocol_version": cfg["protocol_version"],
                 "protocol_status": cfg["status"], "numpy": np.__version__, "pandas": pd.__version__,
                 "rows_loaded": len(df)}
        D.save_subset(sub, rep, ROOT / "data" / "subset_question_ids.csv", ROOT / "data" / "subset_manifest.json", extra)
        print("wrote data/subset_question_ids.csv (ids only) and data/subset_manifest.json")
    print("\nCYCLE 1 DONE. Paste this whole output into code_understanding.md (Cycle 1 section).")
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == "__main__":
    main()
