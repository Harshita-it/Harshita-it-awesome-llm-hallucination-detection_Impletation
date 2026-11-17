"""Token-level scores (protocol Section 5): teacher-forced log-probabilities and entropies."""
import math

import numpy as np


def teacher_forced_token_scores(model, prompt_ids, answer_ids):
    """Score `answer_ids` given `prompt_ids` with ONE forward pass on raw logits (no logit processors).

    prompt_ids, answer_ids: 1-D LongTensors. answer_ids may end with the end-of-turn token.
    Returns dict of CPU float32/long tensors, one entry per answer token:
      logprob  : log p(token | prefix)
      entropy  : Shannon entropy (nats) of the full next-token distribution at that step
      argmax   : most likely token id at that step (for greedy decoding it should equal the token)
      vocab    : size of the logit vector (model vocabulary dimension)
    """
    import torch

    with torch.no_grad():
        device = model.device
        n_prompt = int(prompt_ids.shape[0])
        n_ans = int(answer_ids.shape[0])
        input_ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0).to(device)
        logits = model(input_ids=input_ids).logits[0]          # [L, V]
        pred = logits[n_prompt - 1 : n_prompt + n_ans - 1].float()  # logits that predict answer tokens
        logp = torch.log_softmax(pred, dim=-1)                  # float32
        ans = answer_ids.to(device).unsqueeze(1)
        token_lp = logp.gather(1, ans).squeeze(1)
        entropy = -(logp.exp() * logp).sum(dim=-1)
        return {
            "logprob": token_lp.cpu(),
            "entropy": entropy.cpu(),
            "argmax": pred.argmax(dim=-1).cpu(),
            "vocab": int(pred.shape[-1]),
        }


def aggregate_scores(token_logprobs, token_entropies) -> dict:
    """Aggregators from the protocol. Inputs: sequences of per-token log-probs and entropies
    for the tokens being scored. Higher = more uncertain for every returned value."""
    lp = np.asarray(token_logprobs, dtype=np.float64)
    ent = np.asarray(token_entropies, dtype=np.float64)
    if lp.size == 0:
        nan = float("nan")
        return {"mean_nll": nan, "sum_nll": nan, "max_nll": nan, "mean_entropy": nan, "max_entropy": nan, "n_tokens": 0}
    nll = -lp
    return {
        "mean_nll": float(nll.mean()),
        "sum_nll": float(nll.sum()),
        "max_nll": float(nll.max()),      # = -log(min token probability)
        "mean_entropy": float(ent.mean()),
        "max_entropy": float(ent.max()),
        "n_tokens": int(lp.size),
    }


def worked_example_from_review():
    """The review's illustrative numbers: token probabilities 0.80, 0.60, 0.90."""
    probs = [0.80, 0.60, 0.90]
    lps = [math.log(p) for p in probs]
    return {
        "joint_probability": float(np.prod(probs)),
        "mean_nll": float(-np.mean(lps)),
    }
