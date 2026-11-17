#!/usr/bin/env python
"""Prompt 14: write results/final/results_discussion_outline.md from the REAL final outputs (no new analysis).
Run only after scripts/06_evaluate.py --mode final --frozen has produced results/final/. Mechanisms and alternative explanations are
listed as things to check; [AUTHOR] marks the parts only you can write after looking at the evidence."""
import json

import pandas as pd

from _common import ROOT

from uncertainty.report import build_outline

if __name__ == "__main__":
    fin = ROOT / "results" / "final"
    if not (fin / "summary.json").exists():
        raise SystemExit("results/final/summary.json not found: run scripts/06_evaluate.py --mode final --frozen first. "
                         "This tool never interprets pilot or synthetic numbers.")
    S = json.load(open(fin / "summary.json"))
    if S.get("mode") != "final" or S.get("rq3") is None:
        raise SystemExit("summary.json is not from a final evaluation.")

    def opt_csv(name):
        p = fin / name
        return pd.read_csv(p) if p.exists() and p.stat().st_size > 5 else None

    def opt_json(p):
        return json.load(open(p)) if p.exists() else None

    md = build_outline(S, pd.read_csv(fin / "table2.csv"), pd.read_csv(fin / "table3.csv"), pd.read_csv(fin / "table4.csv"),
                       opt_csv("table6.csv"), opt_json(fin / "extension_E1.json"), opt_csv("table5.csv"),
                       opt_json(ROOT / "results" / "pilot" / "handcheck_agreement.json"))
    out = fin / "results_discussion_outline.md"
    out.write_text(md)
    print(md)
    print(f"\nwrote {out}")
