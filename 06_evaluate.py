#!/usr/bin/env python
"""Cycle 6/7: audit, then evaluation per experiment_protocol.md. [TEST REQUIRED on real data]
  python scripts/06_evaluate.py --mode pilot                       # pilot rows only; writes results/pilot/
  python scripts/06_evaluate.py --mode final --frozen              # test rows; needs git tag protocol-v1; writes results/final/
The pre-evaluation audit runs FIRST and the evaluation refuses to continue on any ERROR (--skip-audit overrides and is recorded).
Options: --n-boot N, --ignore-balance, --skip-tag-check (logged), --skip-audit (logged), --no-figs"""
import argparse
import json

import pandas as pd

from _common import ROOT, audit_from_disk, code_digest, freeze_record, git_commit, git_tag_exists, load_cfg, sha256_file

from uncertainty.audit import format_report
from uncertainty.evaluation import run_analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "final"], required=True)
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--skip-tag-check", action="store_true")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--ignore-balance", action="store_true")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg()
    notes = []
    if a.mode == "final":
        if not a.frozen:
            raise SystemExit("final mode reads the TEST rows: pass --frozen only after experiment_protocol.md Section 14 is complete.")
        if not git_tag_exists("protocol-v1"):
            if not a.skip_tag_check:
                raise SystemExit("git tag protocol-v1 not found. Freeze first: python scripts/11_freeze_protocol.py --freeze")
            notes.append("WARNING: evaluated WITHOUT the protocol-v1 git tag (--skip-tag-check)")
    out = ROOT / "results" / ("pilot" if a.mode == "pilot" else "final")
    out.mkdir(parents=True, exist_ok=True)

    rep = audit_from_disk(a.mode, cfg, strict_freeze=not a.skip_tag_check)
    print(format_report(rep))
    json.dump(rep, open(out / "audit_report.json", "w"), indent=2, default=str)
    if not rep["passed"]:
        if not a.skip_audit:
            raise SystemExit("\nAudit FAILED: no metric was computed. Fix the ERRORs above (or send the audit output for debugging).")
        notes.append("WARNING: evaluated although the audit FAILED (--skip-audit)")

    pq = ROOT / "results" / "per_question.csv"
    rec = freeze_record()
    prov = {"input_file": "results/per_question.csv", "input_sha256": sha256_file(pq), "git_commit": git_commit(),
            "protocol_tag": "protocol-v1" if git_tag_exists("protocol-v1") else None, "freeze_sha256": rec["sha256"] if rec else None,
            "code_sha256": code_digest(), "audit_passed": rep["passed"], "audit_overridden": bool(a.skip_audit and not rep["passed"])}
    s = run_analysis(pd.read_csv(pq), cfg, a.mode, out, n_boot=a.n_boot, make_figs=not a.no_figs, ignore_balance=a.ignore_balance, provenance=prov)
    s["flags"] = notes + s["flags"]
    json.dump(s, open(out / "summary.json", "w"), indent=2, default=str)
    print(f"\nmode={a.mode} | n_eval={s['n_eval']} (wrong {s['n_wrong']}, correct {s['n_correct']}) | bootstrap {s['bootstrap']}")
    print(pd.read_csv(out / "table2.csv")[["score", "auroc", "auroc_lo", "auroc_hi", "auprc", "ece"]].round(3).to_string(index=False))
    print("RQ1:", json.dumps(s["rq1"], default=str))
    if s["rq3"]:
        print("RQ3:", json.dumps(s["rq3"], default=str))
    for f in s["flags"]:
        print("FLAG:", f)
    print(f"outputs in {out} (results.csv carries the input hash, commit and freeze hash on every row)")
    if a.mode == "pilot":
        print("PILOT = exploratory sanity check on calibration-part questions. Do not report pilot numbers as results.")


if __name__ == "__main__":
    main()
