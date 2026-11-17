#!/usr/bin/env python
"""Cycle 5: cluster the 10 samples with DeBERTa-large-MNLI and store the cluster ids. Resumable. [TEST REQUIRED]

  python scripts/03_semantic_entropy.py --split pilot
Output: results/raw/semantic.jsonl (cluster ids, NLI call counts, seconds)."""
import argparse
import time

from _common import GEN, RAW, SEM, load_cfg, require_test_permission

from uncertainty.io_utils import append_jsonl, done_ids, read_jsonl, repair_jsonl
from uncertainty.semantic_entropy import NLIEntailment, cluster_samples, semantic_entropy_from_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["pilot", "calibration", "test", "all"], required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    require_test_permission(args.split, args.allow_test)
    import torch
    cfg = load_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repair_jsonl(SEM)
    gens = read_jsonl(GEN)
    if args.split == "pilot":
        gens = [g for g in gens if g["is_pilot"]]
    elif args.split in ("calibration", "test"):
        gens = [g for g in gens if g["split"] == args.split]
    done = done_ids(SEM)
    todo = sorted([g for g in gens if g["question_id"] not in done], key=lambda g: g["pool_index"])
    if args.limit:
        todo = todo[: args.limit]
    print(f"split={args.split} | generations available {len(gens)} | to cluster now {len(todo)} | device {device}")
    nli = NLIEntailment(cfg["semantic_entropy"]["nli_model"], device)
    print("NLI label map:", nli.id2label, "| entailment index:", nli.entail_idx)
    t0 = time.time()
    for k, g in enumerate(todo, 1):
        t1 = time.time()
        ids, calls = cluster_samples(g["question"], g["samples"], nli)
        if device == "cuda":
            torch.cuda.synchronize()
        append_jsonl(SEM, {"question_id": g["question_id"], "cluster_ids": ids, "nli_calls": calls,
                           "seconds_nli": round(time.time() - t1, 4)})
        if k % 10 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"  {k}/{len(todo)} | {el / k:.2f} s/question | ETA {(len(todo) - k) * el / k / 60:.1f} min")
    recs = [r for r in read_jsonl(SEM)]
    if recs and args.split != "test":
        import numpy as np
        nc = np.array([len(set(r["cluster_ids"])) for r in recs])
        print("\nclusters per question (distribution): ", {int(k): int((nc == k).sum()) for k in np.unique(nc)})
        print("questions where all 10 samples form ONE cluster:", int((nc == 1).sum()), "of", len(nc),
              "(instruct models may show low diversity; report this)")
        print("mean NLI directional calls per question:", round(float(np.mean([r['nli_calls'] for r in recs])), 2))
    print("SEMANTIC ENTROPY DONE")


if __name__ == "__main__":
    main()
