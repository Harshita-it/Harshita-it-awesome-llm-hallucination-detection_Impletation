#!/usr/bin/env python
"""Cycle 2: greedy answer + teacher-forced token scores + 10 samples, per question. Resumable. [TEST REQUIRED]

  python scripts/02_generate.py --split pilot            # first: the 50-question pilot (calibration part only)
  python scripts/02_generate.py --split calibration      # rest of the calibration part
  python scripts/02_generate.py --split test --allow-test  # only after the protocol is frozen
Options: --limit N (debug), --allow-cpu (very slow, debugging only)
Output: results/raw/generations.jsonl (local only: contains question text and gold answers; git-ignored)."""
import argparse
import sys
import time

from pathlib import Path

from _common import GEN, RAW, ROOT, git_commit, load_cfg, load_subset_with_text, require_test_permission, select_rows

from uncertainty.generation import generate_record, load_model_and_tokenizer
from uncertainty.io_utils import append_jsonl, done_ids, read_jsonl, repair_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["pilot", "calibration", "test", "all"], required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--out", default=None, help="alternative output file (e.g. for the reproducibility rerun)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else GEN
    require_test_permission(args.split, args.allow_test)

    import torch
    cfg = load_cfg()
    device = "cuda" if torch.cuda.is_available() else None
    if device is None:
        if not args.allow_cpu:
            raise SystemExit("No GPU. In Colab: Runtime > Change runtime type > T4 GPU (or use --allow-cpu for debugging).")
        device = "cpu"
    RAW.mkdir(parents=True, exist_ok=True)
    n_valid = repair_jsonl(out)
    sub = select_rows(load_subset_with_text(cfg), args.split)
    done = done_ids(out)
    todo = [r for r in sub.itertuples() if r.question_id not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"split={args.split} | selected {len(sub)} | already done {len(sub) - len([r for r in sub.itertuples() if r.question_id not in done])}"
          f" | to run now {len(todo)} | protocol {cfg['protocol_version']} ({cfg['status']})")
    tok, model = load_model_and_tokenizer(cfg["model"]["name"], device)
    import transformers
    append_jsonl(RAW / "meta_runs.jsonl", {"stage": "generate", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "split": args.split,
                                           "n_todo": len(todo), "git_commit": git_commit(), "torch": torch.__version__,
                                           "transformers": transformers.__version__,
                                           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                                           "protocol_status": cfg["status"]})
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        rec = generate_record(model, tok, r.question, cfg, int(r.pool_index))
        rec.update({"question_id": r.question_id, "pool_index": int(r.pool_index), "split": r.split, "is_pilot": bool(r.is_pilot),
                    "question": r.question, "gold_value": r.gold_value, "normalized_aliases": list(r.normalized_aliases)})
        append_jsonl(out, rec)
        if k % 10 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"  {k}/{len(todo)} | {el / k:.2f} s/question | ETA {(len(todo) - k) * el / k / 60:.1f} min")
    recs = read_jsonl(out)
    if recs:
        per_q = sum(r["seconds_greedy"] + r["seconds_samples"] for r in recs) / len(recs)
        print(f"\nmean generation seconds per question over {len(recs)} stored questions: {per_q:.2f}")
        print(f"projected generation time for the pool of {cfg['dataset']['pool_size']}: {per_q * cfg['dataset']['pool_size'] / 3600:.2f} h "
              f"(NLI time comes on top; cap before using the 400 fallback: {cfg['compute']['cap_gpu_hours_before_fallback']} GPU-hours)")
    print("GENERATION DONE (valid lines before this run:", n_valid, ")")


if __name__ == "__main__":
    main()
