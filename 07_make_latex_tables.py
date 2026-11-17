#!/usr/bin/env python
"""Write LaTeX tables from results/final/*.csv into paper/tables/ (no number is typed by hand)."""
from _common import ROOT

from uncertainty.latex_tables import make_tables

if __name__ == "__main__":
    if not (ROOT / "results" / "final" / "table2.csv").exists():
        raise SystemExit("results/final/table2.csv not found: run scripts/06_evaluate.py --mode final --frozen first.")
    made = make_tables(ROOT / "results" / "final", ROOT / "paper" / "tables")
    print("wrote:", made, "-> in main.tex replace the placeholder tables by \\input{tables/<name>.tex}; figures are in results/final/figs/")
