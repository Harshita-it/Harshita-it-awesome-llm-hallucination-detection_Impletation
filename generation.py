"""Cycle 2: greedy answer + teacher-forced token scores + sampled answers for ONE question.

Needs torch/transformers (imported lazily). The helper functions at the top are pure Python and are unit-tested
without a GPU; `generate_record` itself runs the model and is first exercised by scripts/02_generate.py [TEST REQUIRED].
"""
import time

from .textnorm import extract_short_answer


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


def token_char_spans(decode_fn, ids):
    """Character span (start, end) of each token in the decoded text, from decoding growing prefixes.
    decode_fn(list_of_ids) -> str. Special tokens decode to '' (zero-length span). Approximate for
    multi-byte characters split across tokens (documented limitation)."""
    spans, prev = [], 0
    for i in range(len(ids)):
        cur = len(decode_fn(list(ids[: i + 1])))
        spans.append((prev, cur))
        prev = cur
    return spans


def answer_span_mask(spans, text, short_answer):
    """True for tokens overlapping the extracted short answer inside `text`. All False if the answer is not found."""
    if not short_answer:
        return [False] * len(spans)
    start = text.find(short_answer)
    if start < 0:
        return [False] * len(spans)
    end = start + len(short_answer)
    return [bool(e > s and s < end and e > start) for (s, e) in spans]


def load_model_and_tokenizer(model_name, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    except Exception:  # noqa: BLE001  (newer transformers may prefer `dtype`)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model.to(device).eval()
    return tok, model


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_record(model, tok, question, cfg, pool_index):
    """Run the frozen generation protocol for one question. Returns a JSON-serializable dict."""
    import torch

    from .prompting import prompt_input_ids
    from .scoring import teacher_forced_token_scores
    from .seed import seed_for_question, set_seed

    g, s = cfg["generation"]["greedy"], cfg["generation"]["sampling"]
    pids = prompt_input_ids(tok, question).to(model.device)
    attn = torch.ones_like(pids).unsqueeze(0)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = eos_id_set(tok, model)
    n_prompt = int(pids.shape[0])

    # --- greedy answer (deterministic) and its teacher-forced scores
    _sync(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=pids.unsqueeze(0), attention_mask=attn, do_sample=False,
                             temperature=None, top_p=None, top_k=None, repetition_penalty=g["repetition_penalty"],
                             max_new_tokens=g["max_new_tokens"], pad_token_id=pad)
    gen = out[0, n_prompt:].cpu()
    text = tok.decode(gen, skip_special_tokens=True)
    tf = teacher_forced_token_scores(model, pids.cpu(), gen)
    _sync(); t_greedy = time.time() - t0

    ids = [int(x) for x in gen]
    is_eos = [i in eos for i in ids]
    short = extract_short_answer(text)
    decode_fn = lambda x: tok.decode(x, skip_special_tokens=True)  # noqa: E731
    spans = token_char_spans(decode_fn, ids)
    mask = answer_span_mask(spans, text, short)
    truncated = (len(ids) >= g["max_new_tokens"]) and not (len(ids) > 0 and is_eos[-1])

    # --- sampled answers (seeded per question; pure sampling, no truncation)
    set_seed(seed_for_question(s["seed_base"], pool_index))
    _sync(); t1 = time.time()
    with torch.no_grad():
        outs = model.generate(input_ids=pids.unsqueeze(0), attention_mask=attn, do_sample=True,
                              temperature=s["temperature"], top_k=s["top_k"], top_p=s["top_p"],
                              repetition_penalty=s["repetition_penalty"], max_new_tokens=s["max_new_tokens"],
                              num_return_sequences=s["num_samples"], pad_token_id=pad)
    _sync(); t_samples = time.time() - t1
    samples = [tok.decode(o[n_prompt:], skip_special_tokens=True) for o in outs]

    return {
        "greedy_text": text, "short_answer": short, "token_ids": ids,
        "tokens": tok.convert_ids_to_tokens(ids),
        "token_logprob": [float(x) for x in tf["logprob"]], "token_entropy": [float(x) for x in tf["entropy"]],
        "is_eos": is_eos, "span_mask": mask, "truncated": bool(truncated),
        "argmax_match_rate": float((tf["argmax"] == gen).float().mean()) if len(ids) else float("nan"),
        "logit_vector_size": tf["vocab"], "samples": samples,
        "seconds_greedy": round(t_greedy, 4), "seconds_samples": round(t_samples, 4),
    }
