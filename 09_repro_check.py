#!/usr/bin/env python
"""Compare two generation runs on the same questions (greedy answers, token log-probs, samples).
Usage: python scripts/02_generate.py --split pilot --limit 5 --out results/raw/generations_rerun.jsonl
       python scripts/09_repro_check.py results/raw/generations.jsonl results/raw/generations_rerun.jsonl"""
import json
import sys

from _common import ROOT

from uncertainty.io_utils import read_jsonl
from uncertainty.repro import compare_generation_runs

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    res = compare_generation_runs(read_jsonl(sys.argv[1]), read_jsonl(sys.argv[2]))
    print(json.dumps(res, indent=2))
    print("Greedy answers must be identical; log-prob difference within tolerance; identical samples are expected with the same seed "
          "but GPU non-determinism may break this (report it; stored samples are never regenerated).")
