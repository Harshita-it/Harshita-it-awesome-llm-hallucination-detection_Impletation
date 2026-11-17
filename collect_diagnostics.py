#!/usr/bin/env python
"""Collect the evidence needed to debug a failure, safely, into logs/diagnostics.txt (Prompt 11).

Usage (after a failure):
  python scripts/collect_diagnostics.py --traceback-file my_traceback.txt --command "python scripts/02_generate.py --split pilot"
  (save the failing console output to a text file first; you can also add --log logs/env_test_output.txt more than once)
It never prints environment variables and redacts token-like strings. Read the file once before sharing it."""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from _common import ROOT, git_commit, load_cfg

from uncertainty.diagnostics import redact, tail

PKGS = ["torch", "transformers", "datasets", "accelerate", "numpy", "scipy", "scikit-learn", "pandas", "PyYAML", "matplotlib"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traceback-file", default=None)
    ap.add_argument("--command", default="(not given)")
    ap.add_argument("--expected", default="(not given)")
    ap.add_argument("--log", action="append", default=[])
    a = ap.parse_args()
    L = [f"DIAGNOSTICS | {time.strftime('%Y-%m-%d %H:%M:%S')}", f"failing command : {a.command}", f"expected        : {a.expected}",
         f"python          : {sys.version.split()[0]} | {platform.platform()}", f"git commit      : {git_commit()}"]
    try:
        cfg = load_cfg()
        L.append(f"protocol        : {cfg['protocol_version']} ({cfg['status']})")
    except Exception as e:  # noqa: BLE001
        L.append(f"protocol        : could not load config ({type(e).__name__}: {e})")
    L.append("\npackages:")
    for p in PKGS:
        try:
            L.append(f"  {p:<14}{metadata.version(p)}")
        except metadata.PackageNotFoundError:
            L.append(f"  {p:<14}MISSING")
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi], capture_output=True, text=True, timeout=30).stdout.splitlines()[:12]
        L += ["\nnvidia-smi (first lines):"] + ["  " + x for x in out]
    else:
        L.append("\nnvidia-smi: not found (no NVIDIA GPU visible)")
    try:
        import torch
        L.append(f"torch.cuda.is_available(): {torch.cuda.is_available()} | torch CUDA build: {torch.version.cuda}")
    except Exception as e:  # noqa: BLE001
        L.append(f"torch import failed: {type(e).__name__}: {e}")
    L.append("\nfiles (sizes only, no contents):")
    for rel in ["data/subset_question_ids.csv", "data/subset_manifest.json", "results/raw/generations.jsonl", "results/raw/semantic.jsonl",
                "results/per_question.csv", "results/pilot/summary.json", "results/final/summary.json"]:
        p = ROOT / rel
        if p.exists():
            n = sum(1 for _ in open(p, errors="replace")) if p.suffix in (".jsonl", ".csv") else "-"
            L.append(f"  {rel:<36}{p.stat().st_size:>10} bytes | lines {n}")
        else:
            L.append(f"  {rel:<36}missing")
    if a.traceback_file:
        L += ["\n=== traceback / observed output (from file) ===", tail(a.traceback_file, 200)]
    for lg in a.log:
        L += [f"\n=== last 60 lines of {lg} ===", tail(lg, 60)]
    text = redact("\n".join(L))
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "diagnostics.txt").write_text(text)
    print(text)
    print("\nSaved to logs/diagnostics.txt. Read it once, then paste it to Claude together with the code of the failing function.")


if __name__ == "__main__":
    main()
