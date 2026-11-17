#!/usr/bin/env python
"""Pre-registered secondary extension E1 (length-adjusted gate score). Uses TEST rows, so it has the same guards as the final evaluation.
  python scripts/10_extension_length_adjusted.py --frozen     [TEST REQUIRED]
Runs only if the length confound is present by the frozen rule; otherwise it records 'not run' and why."""
import argparse
import json

import pandas as pd

from _common import ROOT, git_tag_exists, load_cfg

from uncertainty.extensions import run_length_adjusted_extension

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--skip-tag-check", action="store_true")
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args()
    if not a.frozen:
        raise SystemExit("E1 reads the TEST rows: pass --frozen only after the protocol is frozen and tagged.")
    if not git_tag_exists("protocol-v1") and not a.skip_tag_check:
        raise SystemExit("git tag protocol-v1 not found.")
    res = run_length_adjusted_extension(pd.read_csv(ROOT / "results" / "per_question.csv"), load_cfg(), ROOT / "results" / "final", a.n_boot)
    print(json.dumps(res, indent=2, default=float))
