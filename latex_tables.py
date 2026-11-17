"""Turn results/final/*.csv into LaTeX tables (so no number in the paper is typed by hand)."""
from pathlib import Path

import pandas as pd


def _ci(v, lo, hi, d=3):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def make_tables(table_dir, out_dir):
    table_dir, out_dir = Path(table_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t2 = pd.read_csv(table_dir / "table2.csv")
    L = ["\\begin{table}[t]", "\\centering\\small\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Detection of wrong answers on the test split: AUROC and AUPRC with 95\\% bootstrap intervals, and ECE after Platt scaling.}",
         "\\label{tab:main}", "\\begin{tabular}{@{}lccc@{}}", "\\toprule", "Score & AUROC & AUPRC & ECE \\\\ \\midrule"]
    for _, r in t2.iterrows():
        ece = "--" if pd.isna(r["ece"]) else f"{r['ece']:.3f}"
        name = str(r["score"]).replace("&", "\\&")
        L.append(f"{name} & {_ci(r['auroc'], r['auroc_lo'], r['auroc_hi'])} & "
                 f"{_ci(r['auprc'], r['auprc_lo'], r['auprc_hi'])} & {ece} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out_dir / "table_main.tex").write_text("\n".join(L) + "\n")

    p4 = table_dir / "table4.csv"
    if p4.exists():
        t4 = pd.read_csv(p4)
        name = {"token-only": "Token score only", "primary": "Cascade ($q{=}0.5$)", "SE-for-all": "SE for all questions"}
        L = ["\\begin{table}[t]", "\\centering\\small\\setlength{\\tabcolsep}{4pt}",
             "\\caption{Cascade at the primary operating point versus the two references (counted extra generations per question).}",
             "\\label{tab:cascade}", "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
             "System & AUROC & Escal. & Extra gen. \\\\ \\midrule"]
        for role in ("token-only", "primary", "SE-for-all"):
            r = t4[t4.role == role].iloc[0]
            L.append(f"{name[role]} & {_ci(r['auroc'], r['auroc_lo'], r['auroc_hi'])} & {100 * r['escalated_fraction']:.0f}\\% & "
                     f"{r['extra_generations_per_question']:.1f} \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (out_dir / "table_cascade.tex").write_text("\n".join(L) + "\n")

    t3 = pd.read_csv(table_dir / "table3.csv")
    L = ["\\begin{table}[t]", "\\centering\\small\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Aggregation ablation (AUROC, 95\\% CI) and the three pre-specified contrasts (98.3\\% CI).}",
         "\\label{tab:ablation}", "\\begin{tabular}{@{}lcc@{}}", "\\toprule", "Variant / contrast & Value [CI] & Matters \\\\ \\midrule"]
    for _, r in t3.iterrows():
        m = "" if pd.isna(r["matters"]) else ("yes" if bool(r["matters"]) else "no")
        L.append(f"{str(r['name']).replace('_', ' ')} & {_ci(r['auroc'], r['lo'], r['hi'])} & {m} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out_dir / "table_ablation.tex").write_text("\n".join(L) + "\n")

    p6 = table_dir / "table6.csv"
    if p6.exists() and p6.stat().st_size > 5:
        t6 = pd.read_csv(p6)
        L = ["\\begin{table}[t]", "\\centering\\small", "\\caption{Sensitivity analyses (AUROC, 95\\% CI).}", "\\label{tab:sens}",
             "\\begin{tabular}{@{}lc@{}}", "\\toprule", "Analysis & AUROC [CI] \\\\ \\midrule"]
        for _, r in t6.iterrows():
            L.append(f"{r['analysis']} & {_ci(r['auroc'], r['lo'], r['hi'])} \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (out_dir / "table_sensitivity.tex").write_text("\n".join(L) + "\n")
    return sorted(p.name for p in out_dir.glob("*.tex"))
