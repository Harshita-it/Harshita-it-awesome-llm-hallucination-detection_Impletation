#!/usr/bin/env python
"""Hand-check sheets. [TEST REQUIRED]
  python scripts/05_handcheck.py --make-labelcheck           -> results/raw/handcheck_sheet.csv (30 pilot answers; NO scores shown)
  (fill the column human_correct with yes/no by hand, WITHOUT looking at any uncertainty score)
  python scripts/05_handcheck.py --score-labelcheck          -> agreement with the automatic label (target >= 0.90)
  python scripts/05_handcheck.py --make-erroranalysis --frozen        (after the protocol tag; uses TEST rows)
  python scripts/05_handcheck.py --summarize-erroranalysis --frozen   (after filling the `category` column)"""
import argparse
import csv
import json

import pandas as pd

from _common import GEN, RAW, ROOT, SEM, git_tag_exists, load_cfg

from uncertainty.handcheck import (ERROR_CATEGORIES, agreement, parse_human_label, select_error_analysis,
                                   select_for_labelcheck)
from uncertainty.io_utils import read_jsonl
from uncertainty.textnorm import is_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-labelcheck", action="store_true")
    ap.add_argument("--score-labelcheck", action="store_true")
    ap.add_argument("--make-erroranalysis", action="store_true")
    ap.add_argument("--summarize-erroranalysis", action="store_true")
    ap.add_argument("--show-clusters", type=int, default=0, help="print the samples and cluster ids of N pilot questions for manual validation")
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--skip-tag-check", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg()
    hc = cfg["label"]["hand_check"]
    sheet = RAW / "handcheck_sheet.csv"
    gens = {g["question_id"]: g for g in read_jsonl(GEN)}

    if a.show_clusters:
        sems = {r["question_id"]: r for r in read_jsonl(SEM)}
        shown = 0
        for g in sorted([g for g in gens.values() if g["is_pilot"]], key=lambda g: g["pool_index"]):
            if g["question_id"] not in sems or shown >= a.show_clusters:
                continue
            ids = sems[g["question_id"]]["cluster_ids"]
            print(f"\nQ: {g['question']}\n   gold: {g['gold_value']} | greedy: {g['short_answer']!r} | clusters: {len(set(ids))}")
            for c in sorted(set(ids)):
                print(f"   cluster {c}: " + " ; ".join(repr(s.strip()[:40]) for s, i in zip(g["samples"], ids) if i == c))
            shown += 1
        print("\nFor each question note: wrong MERGE (different meanings in one cluster) or wrong SPLIT (same meaning in two clusters).")

    if a.make_labelcheck:
        pilot = [g for g in gens.values() if g["is_pilot"]]
        chosen = select_for_labelcheck(pilot, hc["n"], hc["seed"])
        if len(chosen) < hc["n"]:
            print(f"WARNING: only {len(chosen)} pilot answers available")
        with open(sheet, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question_id", "question", "model_answer", "gold_value", "gold_aliases", "human_correct"])
            for g in chosen:
                w.writerow([g["question_id"], g["question"], g["short_answer"], g["gold_value"],
                            " | ".join(g["normalized_aliases"][:8]), ""])
        print(f"wrote {sheet} ({len(chosen)} rows). Fill human_correct with yes/no. Do NOT look at uncertainty scores while doing it.")

    if a.score_labelcheck:
        df = pd.read_csv(sheet, dtype=str).fillna("")
        auto = [is_correct(gens[q]["greedy_text"], gens[q]["normalized_aliases"]) for q in df["question_id"]]
        hum = [parse_human_label(x) for x in df["human_correct"]]
        res = agreement(auto, hum)
        res["target"] = hc["min_agreement"]
        res["meets_target"] = bool(res["agreement"] >= hc["min_agreement"]) if res["n_compared"] else False
        print(json.dumps(res, indent=2))
        for q, au, hu in zip(df["question_id"], auto, hum):
            if hu is not None and au != hu:
                print("DISAGREEMENT:", q, "| answer:", gens[q]["short_answer"], "| gold:", gens[q]["gold_value"], "| auto_correct:", au, "| human_correct:", hu)
        out = ROOT / "results" / "pilot"
        out.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out / "handcheck_agreement.json", "w"), indent=2)

    if a.make_erroranalysis or a.summarize_erroranalysis:
        if not a.frozen or not (a.skip_tag_check or git_tag_exists()):
            raise SystemExit("Error analysis uses TEST rows: needs --frozen and the git tag protocol-v1 (or --skip-tag-check).")
        path = RAW / "error_analysis_sheet.csv"
        if a.make_erroranalysis:
            t = pd.read_csv(ROOT / "results" / "per_question.csv")
            t = t[(t["split"] == "test") & t["SE10"].notna()]
            wrong, right = select_error_analysis(t.to_dict("records"), 30)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["group", "question_id", "question", "model_answer", "gold_value", "n_clusters", "category"])
                for grp, rows in (("confident_error", wrong), ("uncertain_correct", right)):
                    for r in rows:
                        g = gens[r["question_id"]]
                        w.writerow([grp, r["question_id"], g["question"], g["short_answer"], g["gold_value"], r["n_clusters"], ""])
            print(f"wrote {path}. Fill `category` with one of: {ERROR_CATEGORIES}")
        else:
            df = pd.read_csv(path, dtype=str).fillna("")
            bad = df[~df["category"].isin(ERROR_CATEGORIES)]
            if len(bad):
                raise SystemExit(f"{len(bad)} rows have a missing/invalid category")
            out = ROOT / "results" / "final"
            out.mkdir(parents=True, exist_ok=True)
            df.groupby(["group", "category"]).size().reset_index(name="count").to_csv(out / "table5.csv", index=False)
            print(df.groupby(["group", "category"]).size())


if __name__ == "__main__":
    main()
