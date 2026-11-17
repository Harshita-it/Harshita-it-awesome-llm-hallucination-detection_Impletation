#!/usr/bin/env python
"""Cycle 4: merge generations + semantic entropy into results/per_question.csv (no question text, no gold answers).
Prints COUNTS only (class counts are allowed before the protocol tag; scores of test rows are never printed)."""
import numpy as np

from _common import GEN, ROOT, SEM, load_cfg

from uncertainty.io_utils import read_jsonl
from uncertainty.table import build_table


def main():
    gens, sems = read_jsonl(GEN), read_jsonl(SEM)
    if not gens:
        raise SystemExit("results/raw/generations.jsonl is empty: run scripts/02_generate.py first")
    df = build_table(gens, sems)
    out = ROOT / "results" / "per_question.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} | rows {len(df)} | with semantic entropy {int(df['SE10'].notna().sum())}")
    print("rows per split:", df["split"].value_counts().to_dict(), "| pilot rows:", int(df["is_pilot"].sum()))
    for sp in ("calibration", "test"):
        d = df[df["split"] == sp]
        if len(d):
            print(f"{sp}: correct {int((d['y_wrong_exact'] == 0).sum())} | wrong {int((d['y_wrong_exact'] == 1).sum())} (exact-match label)")
    nt = df[df["split"] != "test"]            # diagnostics on non-test rows only
    print("non-test diagnostics -> truncated:", int(nt["truncated"].sum()), "| span fallback:", int(nt["span_fallback"].sum()),
          "| zero-token answers:", int((nt["n_tokens"] == 0).sum()),
          "| mean argmax=token rate (greedy, expect about 1.0):", round(float(np.nanmean(nt["argmax_match_rate"])), 4) if len(nt) else "n/a")


if __name__ == "__main__":
    main()
