#!/usr/bin/env python
"""Environment test for the token-uncertainty project (protocol-v1).

Run from the repository root:   python scripts/00_env_test.py
Useful flags: --skip-dataset  --skip-nli  --allow-cpu  --n-questions 5

What it does (each stage is independent and reports PASS / WARN / FAIL):
  1 system        Python, OS, RAM, disk, nvidia-smi
  2 libraries     versions of the packages we use; transformers >= 4.37 for Qwen2.5
  3 gpu           CUDA visible, device name, memory, small fp16 matrix multiply
  4 seed          seeding gives identical random numbers twice
  5 config        configs/protocol_v1.yaml loads and has the key settings
  6 dataset       TriviaQA rc.nocontext validation: fields, size, duplicate question_ids
  7 model         loads Qwen2.5-1.5B-Instruct in fp16; prints generation_config and rendered prompt
  8 greedy        greedy answers + teacher-forced token scores + consistency check vs generate-time scores
  9 sampling      10 samples at T=1 (no truncation); same seed twice gives same samples?
 10 nli           loads deberta-large-mnli; prints label order; 3 hand-made pairs
 11 report        writes results/env_report.json, results/requirements_candidate.txt, results/pip_freeze.txt

Nothing here prints tokens or environment variables. Output is also saved to logs/env_test_output.txt.
"""
import argparse
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = []
CTX = {}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def run_stage(name, fn, *args, **kwargs):
    t0 = time.time()
    print(f"\n=== [{name}] ===")
    try:
        status, detail = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001  (we want to keep going and report)
        status, detail = "FAIL", {"error": f"{type(exc).__name__}: {exc}"}
        print("FAIL:", detail["error"])
        traceback.print_exc(limit=4, file=sys.stdout)
    dt = time.time() - t0
    RESULTS.append({"stage": name, "status": status, "seconds": round(dt, 2), "detail": detail})
    print(f"--> {name}: {status} ({dt:.1f}s)")
    return status


def dist_version(dist_name):
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


# ---------------------------------------------------------------- stages
def stage_system():
    d = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count()}
    print("Python  :", d["python"])
    print("Platform:", d["platform"])
    print("CPUs    :", d["cpu_count"])
    try:
        with open("/proc/meminfo") as f:
            mem_kb = int(f.readline().split()[1])
        d["ram_gb"] = round(mem_kb / 1024 / 1024, 1)
        print("RAM (GB):", d["ram_gb"])
    except Exception:  # noqa: BLE001
        print("RAM     : not readable on this OS")
    total, used, free = shutil.disk_usage(str(ROOT))
    d["disk_free_gb"] = round(free / 1e9, 1)
    print("Disk free (GB):", d["disk_free_gb"])
    smi = shutil.which("nvidia-smi")
    if not smi:
        print("nvidia-smi not found: no NVIDIA GPU visible to this runtime.")
        d["nvidia_smi"] = None
        return "WARN", d
    out = subprocess.run([smi], capture_output=True, text=True, timeout=30).stdout
    print(out)
    d["nvidia_smi"] = out.splitlines()[:12]
    return "PASS", d


def stage_libraries():
    pkgs = [("torch", "torch"), ("transformers", "transformers"), ("datasets", "datasets"),
            ("accelerate", "accelerate"), ("numpy", "numpy"), ("scipy", "scipy"),
            ("scikit-learn", "scikit-learn"), ("pandas", "pandas"), ("PyYAML", "PyYAML"),
            ("matplotlib", "matplotlib")]
    versions, missing = {}, []
    for label, dist in pkgs:
        v = dist_version(dist)
        versions[label] = v
        print(f"{label:<14}{v if v else 'MISSING'}")
        if v is None:
            missing.append(label)
    CTX["versions"] = versions
    status = "PASS"
    notes = []
    tv = versions.get("transformers")
    if tv:
        try:
            from packaging.version import Version
            if Version(tv) < Version("4.37.0"):
                notes.append("transformers < 4.37.0: Qwen2.5 loading fails with KeyError: 'qwen2' (model card). Upgrade.")
                status = "FAIL"
        except Exception:  # noqa: BLE001
            notes.append("could not compare transformers version")
    if missing:
        notes.append(f"missing packages: {missing}")
        status = "FAIL"
    for n in notes:
        print("NOTE:", n)
    return status, {"versions": versions, "notes": notes}


def stage_gpu(allow_cpu):
    import torch
    d = {"torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    print("torch CUDA build :", d["torch_cuda_build"])
    print("CUDA available   :", d["cuda_available"])
    if not torch.cuda.is_available():
        CTX["device"] = "cpu"
        print("No CUDA device. In Colab: Runtime > Change runtime type > T4 GPU, then restart and rerun.")
        return ("WARN" if allow_cpu else "FAIL"), d
    CTX["device"] = "cuda"
    props = torch.cuda.get_device_properties(0)
    d.update({"name": props.name, "total_memory_gb": round(props.total_memory / 1e9, 1),
              "capability": f"{props.major}.{props.minor}", "bf16_supported": bool(torch.cuda.is_bf16_supported())})
    print("GPU              :", d["name"])
    print("GPU memory (GB)  :", d["total_memory_gb"])
    print("Compute capab.   :", d["capability"])
    print("bf16 supported   :", d["bf16_supported"], "(protocol uses fp16)")
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.time()
    c = a @ b
    torch.cuda.synchronize()
    d["fp16_matmul_seconds"] = round(time.time() - t0, 4)
    ok = bool(torch.isfinite(c).all().item())
    print("fp16 matmul 2048x2048 finite:", ok, f"({d['fp16_matmul_seconds']}s, includes warm-up)")
    return ("PASS" if ok else "FAIL"), d


def stage_seed():
    import random
    import numpy as np
    import torch
    from uncertainty.seed import set_seed

    def draw():
        vals = [random.random(), float(np.random.rand()), float(torch.rand(1).item())]
        if torch.cuda.is_available():
            vals.append(float(torch.rand(1, device="cuda").item()))
        return vals

    set_seed(42)
    first = draw()
    set_seed(42)
    second = draw()
    print("first :", first)
    print("second:", second)
    same = first == second
    print("identical after reseeding:", same)
    return ("PASS" if same else "FAIL"), {"identical": same}


def stage_config():
    import yaml
    cfg_path = ROOT / "configs" / "protocol_v1.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    CTX["cfg"] = cfg
    print("protocol_version:", cfg["protocol_version"], "| status:", cfg["status"])
    print("model:", cfg["model"]["name"], "| dataset:", cfg["dataset"]["hf_name"], cfg["dataset"]["config"], cfg["dataset"]["split"])
    print("primary score:", cfg["scoring"]["primary_score"], "| samples:", cfg["generation"]["sampling"]["num_samples"])
    return "PASS", {"protocol_version": cfg["protocol_version"], "status": cfg["status"]}


def stage_dataset(check_test):
    from datasets import load_dataset
    cfg = CTX["cfg"]["dataset"]
    ds = load_dataset(cfg["hf_name"], cfg["config"], split=cfg["split"])
    CTX["ds"] = ds
    print("rows:", len(ds))
    print("columns:", ds.column_names)
    ex = ds[0]
    print("example question   :", ex["question"])
    print("example question_id:", ex["question_id"])
    print("answer.value       :", ex["answer"]["value"])
    print("n aliases          :", len(ex["answer"]["aliases"]), "| first normalized:", ex["answer"]["normalized_aliases"][:5])
    ids = ds["question_id"]
    n_unique = len(set(ids))
    n_dup = len(ids) - n_unique
    print(f"unique question_id: {n_unique} | duplicate rows: {n_dup}")
    qtexts = ds["question"]
    print("unique question strings:", len(set(qtexts)))
    empty = sum(1 for a in ds["answer"] if not a["normalized_aliases"])
    print("rows with no aliases:", empty)
    detail = {"rows": len(ds), "unique_question_id": n_unique, "duplicate_rows": n_dup,
              "unique_question_text": len(set(qtexts)), "rows_without_aliases": empty}
    if check_test:
        te = load_dataset(cfg["hf_name"], cfg["config"], split="test")
        no_ans = sum(1 for a in te.select(range(min(100, len(te))))["answer"] if not a["normalized_aliases"])
        print(f"test split: first 100 rows without aliases = {no_ans}")
        detail["test_first100_without_aliases"] = no_ans
    status = "WARN" if (n_dup > 0 or empty > 0) else "PASS"
    if n_dup > 0:
        print("NOTE: duplicates exist; deduplicate by question_id before sampling (protocol Section 4, step 2).")
    return status, detail


def load_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = CTX["cfg"]["model"]["name"]
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
    except Exception as exc:  # newer transformers may prefer `dtype`
        print("torch_dtype load failed (", type(exc).__name__, "), retrying with dtype=")
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
    model.to(CTX["device"]).eval()
    return tok, model


def stage_model():
    import torch
    from uncertainty.prompting import render_chat_prompt
    tok, model = load_model_and_tokenizer()
    CTX["tok"], CTX["model"] = tok, model
    n_params = sum(p.numel() for p in model.parameters())
    print("parameters (M):", round(n_params / 1e6, 1))
    print("dtype:", next(model.parameters()).dtype, "| device:", next(model.parameters()).device)
    print("tokenizer length:", len(tok), "| model.config.vocab_size:", model.config.vocab_size)
    print("eos_token:", tok.eos_token, tok.eos_token_id, "| generation_config.eos_token_id:", model.generation_config.eos_token_id)
    gc = model.generation_config.to_dict()
    keys = ["do_sample", "temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens", "eos_token_id", "pad_token_id"]
    print("generation_config (selected):", {k: gc.get(k) for k in keys})
    rp = gc.get("repetition_penalty")
    notes = []
    if rp not in (None, 1.0):
        notes.append(f"default repetition_penalty={rp}: would distort generate-time scores; protocol sets 1.0 explicitly")
    rendered = render_chat_prompt(tok, "Who wrote the novel 'Pride and Prejudice'?")
    print("--- rendered prompt (this exact text, incl. any default system message, is part of the protocol) ---")
    print(rendered)
    print("--- end rendered prompt ---")
    if torch.cuda.is_available():
        print("GPU memory allocated (GB):", round(torch.cuda.memory_allocated() / 1e9, 2))
    for n in notes:
        print("NOTE:", n)
    detail = {"params_million": round(n_params / 1e6, 1), "generation_config": {k: gc.get(k) for k in keys}, "notes": notes}
    return ("WARN" if notes else "PASS"), detail


def eos_id_set(tok, model):
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    e = model.generation_config.eos_token_id
    if isinstance(e, int):
        ids.add(e)
    elif isinstance(e, (list, tuple)):
        ids.update(int(x) for x in e)
    return ids


def stage_greedy(n_questions):
    import torch
    from uncertainty.prompting import prompt_input_ids
    from uncertainty.scoring import aggregate_scores, teacher_forced_token_scores
    from uncertainty.textnorm import is_correct

    tok, model, cfg = CTX["tok"], CTX["model"], CTX["cfg"]
    tol = cfg["scoring"]["consistency_check"]["tolerance"]
    max_new = cfg["generation"]["greedy"]["max_new_tokens"]
    eos = eos_id_set(tok, model)

    if "ds" in CTX:
        qs = [(CTX["ds"][i]["question"], CTX["ds"][i]["answer"]["normalized_aliases"], CTX["ds"][i]["answer"]["value"])
              for i in range(n_questions)]
    else:
        qs = [("Who wrote the novel 'Pride and Prejudice'?", ["jane austen", "austen"], "Jane Austen")]
        print("(dataset not loaded: using one fixed question)")

    max_diffs, rows, flags = [], [], []
    for qi, (q, aliases, gold) in enumerate(qs):
        pids = prompt_input_ids(tok, q).to(model.device)
        out = model.generate(input_ids=pids.unsqueeze(0), attention_mask=torch.ones_like(pids).unsqueeze(0),
                             do_sample=False, temperature=None, top_p=None, top_k=None,
                             repetition_penalty=1.0, max_new_tokens=max_new,
                             output_scores=True, return_dict_in_generate=True,
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        gen = out.sequences[0, pids.shape[0]:].cpu()
        text = tok.decode(gen, skip_special_tokens=True)
        tf = teacher_forced_token_scores(model, pids.cpu(), gen)
        # generate-time log-probs from the returned scores
        if len(out.scores) != gen.shape[0]:
            flags.append(f"q{qi}: len(scores)={len(out.scores)} != n generated tokens={gen.shape[0]}")
            continue
        gen_lp = torch.stack([torch.log_softmax(s[0].float(), -1)[gen[t].to(s.device)].cpu() for t, s in enumerate(out.scores)])
        diff = (gen_lp - tf["logprob"]).abs()
        max_diffs.append(float(diff.max()))
        keep = [i for i in range(gen.shape[0]) if int(gen[i]) not in eos]
        agg = aggregate_scores(tf["logprob"][keep].tolist(), tf["entropy"][keep].tolist())
        argmax_rate = float((tf["argmax"] == gen).float().mean())
        correct = is_correct(text, aliases)
        rows.append((qi, text.strip()[:40], gold[:25], correct, gen.shape[0], round(agg["mean_nll"], 3), round(argmax_rate, 2), round(max_diffs[-1], 4)))
        if qi == 0:
            print("\nFirst question, per-token table (EOS tokens are scored here but excluded from aggregates):")
            print(f"{'token':<16}{'id':>8}{'logprob':>10}{'prob':>8}{'entropy':>9}  eos?")
            for t in range(gen.shape[0]):
                lp, h = float(tf["logprob"][t]), float(tf["entropy"][t])
                print(f"{tok.decode(gen[t])!r:<16}{int(gen[t]):>8}{lp:>10.3f}{math.exp(lp):>8.3f}{h:>9.3f}  {'EOS' if int(gen[t]) in eos else ''}")
                if lp > 1e-4 or h < -1e-4 or h > math.log(tf["vocab"]) + 1e-3:
                    flags.append(f"q0 token {t}: value outside valid range (logprob {lp}, entropy {h})")
            print("logit vector size (V):", tf["vocab"])
    print(f"\n{'q':<3}{'answer':<42}{'gold':<27}{'ok?':<6}{'nTok':<6}{'meanNLL':<9}{'argmax=tok':<12}{'max|diff|'}")
    for r in rows:
        print(f"{r[0]:<3}{r[1]!r:<42}{r[2]!r:<27}{str(r[3]):<6}{r[4]:<6}{r[5]:<9}{r[6]:<12}{r[7]}")
    worst = max(max_diffs) if max_diffs else float("nan")
    print(f"\nconsistency check: max |generate-time logprob - teacher-forced logprob| = {worst:.4f} (tolerance {tol}, provisional)")
    for f in flags:
        print("FLAG:", f)
    status = "PASS"
    if flags or not max_diffs or worst > tol:
        status = "WARN"
    return status, {"max_abs_logprob_diff": worst, "tolerance": tol, "flags": flags, "rows": [list(map(str, r)) for r in rows]}


def stage_sampling():
    import torch
    from uncertainty.prompting import prompt_input_ids
    from uncertainty.seed import seed_for_question, set_seed
    from uncertainty.textnorm import extract_short_answer, normalize_answer

    tok, model, cfg = CTX["tok"], CTX["model"], CTX["cfg"]
    sc = cfg["generation"]["sampling"]
    q = CTX["ds"][0]["question"] if "ds" in CTX else "Who wrote the novel 'Pride and Prejudice'?"
    pids = prompt_input_ids(tok, q).to(model.device)

    def sample():
        set_seed(seed_for_question(sc["seed_base"], 0))
        t0 = time.time()
        out = model.generate(input_ids=pids.unsqueeze(0), attention_mask=torch.ones_like(pids).unsqueeze(0),
                             do_sample=True, temperature=sc["temperature"], top_k=sc["top_k"], top_p=sc["top_p"],
                             repetition_penalty=sc["repetition_penalty"], max_new_tokens=sc["max_new_tokens"],
                             num_return_sequences=sc["num_samples"],
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        texts = [tok.decode(o[pids.shape[0]:], skip_special_tokens=True) for o in out]
        return texts, dt

    a, t_a = sample()
    b, t_b = sample()
    print("question:", q)
    for i, t in enumerate(a):
        print(f"  sample {i + 1:>2}: {extract_short_answer(t)!r}")
    uniq = len({normalize_answer(extract_short_answer(t)) for t in a})
    print("distinct normalized answers among 10:", uniq)
    same = a == b
    print("same seed -> identical samples:", same)
    print(f"seconds for 10 samples (one question, rough): {t_a:.2f}, {t_b:.2f}")
    status = "PASS" if same else "WARN"
    if not same:
        print("NOTE: GPU sampling may be non-deterministic across calls; log this and rely on stored samples.")
    return status, {"distinct_answers": uniq, "reproducible": same, "seconds_10_samples": round(t_a, 2)}


def stage_nli():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = CTX["cfg"]["semantic_entropy"]["nli_model"]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(CTX["device"]).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print("id2label (read from the model, do not assume):", id2label)
    print("tokenizer.model_max_length:", tok.model_max_length, "(we set max_length explicitly in calls)")
    tests = [
        ("The cat sat on the mat.", "A cat is on the mat.", "entailment"),
        ("The cat sat on the mat.", "The cat is not on the mat.", "contradiction"),
        ("The cat sat on the mat.", "The cat is hungry.", "neutral"),
    ]
    ok = 0
    for prem, hyp, expected in tests:
        enc = tok(prem, hyp, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits[0].float(), -1).cpu()
        pred = id2label[int(probs.argmax())].lower()
        ok += int(pred == expected)
        print(f"premise={prem!r} hypothesis={hyp!r}\n  expected={expected} predicted={pred} probs={[round(float(p), 3) for p in probs]}")
    CTX["nli_ok"] = ok
    return ("PASS" if ok == len(tests) else "WARN"), {"id2label": id2label, "hand_pairs_as_expected": f"{ok}/{len(tests)}"}


def stage_report():
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    versions = CTX.get("versions", {})
    with open(out_dir / "env_report.json", "w") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": RESULTS}, f, indent=2, default=str)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out_dir / "pip_freeze.txt").write_text(freeze)
    lines = [f"{k}=={v}" for k, v in versions.items() if v]
    (out_dir / "requirements_candidate.txt").write_text(
        "# Versions seen in THIS run. Commit as requirements.txt ONLY if all stages passed.\n" + "\n".join(lines) + "\n")
    print("wrote results/env_report.json, results/pip_freeze.txt, results/requirements_candidate.txt")
    return "PASS", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--skip-nli", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="continue without a GPU (slow; for debugging only)")
    ap.add_argument("--check-test", action="store_true", help="also confirm the TriviaQA test split has no answers")
    ap.add_argument("--n-questions", type=int, default=5)
    args = ap.parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    log_f = open(ROOT / "logs" / "env_test_output.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log_f)
    print("ENVIRONMENT TEST  | started", time.strftime("%Y-%m-%d %H:%M:%S"), "| repo:", ROOT)

    run_stage("1 system", stage_system)
    s = run_stage("2 libraries", stage_libraries)
    if s == "FAIL" and any(v is None for k, v in CTX.get("versions", {}).items() if k in ("torch", "transformers")):
        print("\nCore libraries missing: cannot continue. Install them (see README) and rerun.")
    else:
        g = run_stage("3 gpu", stage_gpu, args.allow_cpu)
        run_stage("4 seed", stage_seed)
        c = run_stage("5 config", stage_config)
        if c == "PASS" and not args.skip_dataset:
            run_stage("6 dataset", stage_dataset, args.check_test)
        if c == "PASS" and (g != "FAIL"):
            m = run_stage("7 model", stage_model)
            if m != "FAIL":
                run_stage("8 greedy+scoring+consistency", stage_greedy, args.n_questions)
                run_stage("9 sampling", stage_sampling)
            if not args.skip_nli:
                run_stage("10 nli", stage_nli)
    run_stage("11 report", stage_report)

    print("\n================ SUMMARY ================")
    for r in RESULTS:
        print(f"{r['stage']:<32}{r['status']:<6}{r['seconds']:>8.1f}s")
    n = {k: sum(1 for r in RESULTS if r["status"] == k) for k in ("PASS", "WARN", "FAIL")}
    print(f"\nENV TEST SUMMARY: {n['PASS']} PASS, {n['WARN']} WARN, {n['FAIL']} FAIL")
    print("Next step: paste this whole output into environment_verification_log.md. Do not edit it.")
    sys.stdout = sys.__stdout__
    log_f.close()
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
