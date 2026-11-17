#!/usr/bin/env python
"""Pre-evaluation audit (alignment, missing values, partitions, metric direction, leakage, comparator consistency). No metric is computed.
  python scripts/12_audit_outputs.py --mode pilot
  python scripts/12_audit_outputs.py --mode final          (needs the freeze; --skip-tag-check downgrades freeze errors to warnings)
Exit code 1 if any ERROR. Writes results/<mode>/audit_report.json. scripts/06_evaluate.py runs this automatically and refuses to continue on ERROR."""
import argparse
import json

from _common import ROOT, load_cfg, audit_from_disk

from uncertainty.audit import format_report

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "final"], required=True)
    ap.add_argument("--skip-tag-check", action="store_true")
    a = ap.parse_args()
    rep = audit_from_disk(a.mode, load_cfg(), strict_freeze=not a.skip_tag_check)
    out = ROOT / "results" / a.mode
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out / "audit_report.json", "w"), indent=2, default=str)
    print(format_report(rep))
    raise SystemExit(0 if rep["passed"] else 1)
