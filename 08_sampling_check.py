#!/usr/bin/env python
"""Statistical check that sampling is NOT silently truncated (top-k/top-p from the model's default generation config).
For a few questions: draw many first-token samples with the protocol settings and compare the observed frequency of the most
likely token with its model probability (z-score), and the observed mass outside the top-20 tokens with the model's tail mass.
Needs a GPU. [TEST REQUIRED]   Usage: python scripts/08_sampling_check.py [--n 2000]"""
import argparse
import math

import torch
from _common import load_cfg

from uncertainty.generation import load_model_and_tokenizer
from uncertainty.prompting import prompt_input_ids
from uncertainty.seed import set_seed

QUESTIONS = ["Who wrote the novel 'Pride and Prejudice'?", "What is the capital city of Australia?",
             "Which element has the chemical symbol Fe?", "In which year did the Titanic sink?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    n = ap.parse_args().n
    cfg = load_cfg()
    s = cfg["generation"]["sampling"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model_and_tokenizer(cfg["model"]["name"], device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"{'question':<46}{'p(top1)':>9}{'freq':>8}{'z':>7}{'tail mass':>11}{'observed':>10}")
    worst = 0.0
    for q in QUESTIONS:
        pids = prompt_input_ids(tok, q).to(model.device)
        with torch.no_grad():
            p = torch.softmax(model(pids.unsqueeze(0)).logits[0, -1].float(), -1)
        top1 = int(p.argmax())
        p1 = float(p[top1])
        tail = float(torch.sort(p, descending=True).values[20:].sum())          # mass outside the 20 most likely tokens
        top20 = set(torch.topk(p, 20).indices.tolist())
        set_seed(12345)
        with torch.no_grad():
            out = model.generate(input_ids=pids.unsqueeze(0).repeat(n, 1), attention_mask=torch.ones(n, pids.shape[0], device=model.device, dtype=torch.long),
                                 do_sample=True, temperature=s["temperature"], top_k=s["top_k"], top_p=s["top_p"],
                                 repetition_penalty=s["repetition_penalty"], max_new_tokens=1, pad_token_id=pad)
        first = out[:, pids.shape[0]].tolist()
        freq = sum(t == top1 for t in first) / n
        obs_tail = sum(t not in top20 for t in first) / n
        z = (freq - p1) / math.sqrt(max(p1 * (1 - p1), 1e-12) / n)
        worst = max(worst, abs(z))
        print(f"{q[:44]:<46}{p1:>9.3f}{freq:>8.3f}{z:>7.2f}{tail:>11.4f}{obs_tail:>10.4f}")
    print(f"\nlargest |z| = {worst:.2f}  -> {'PASS (consistent with untruncated sampling)' if worst < 4 else 'FAIL: sampling looks truncated or temperature differs; do NOT continue'}")
    print("If tail mass is clearly > 0 but observed is 0.0000, top-k/top-p from the default generation config is still active.")


if __name__ == "__main__":
    main()
