"""Cycle 5: semantic entropy (discrete variant) by greedy NLI clustering of sampled answers (protocol Section 7).

`cluster_samples` is pure Python: it takes an `entails(question, a, b) -> bool` function, so it is unit-tested with a fake.
`NLIEntailment` wraps DeBERTa-large-MNLI [TEST REQUIRED on GPU].
"""
import math
from collections import Counter

from .textnorm import extract_short_answer, normalize_answer


def cluster_samples(question, samples, entails):
    """Greedy sequential clustering.
    A sample joins the FIRST existing cluster whose representative (its first member) it matches:
      - identical normalized strings match without an NLI call (documented shortcut), otherwise
      - strict bidirectional entailment: entails(rep, sample) AND entails(sample, rep).
    Empty answers never go through NLI (they only match other empty answers).
    Returns (cluster_ids, n_nli_calls) where n_nli_calls counts directional entailment evaluations."""
    reps, ids, calls = [], [], 0
    for s in samples:
        short = extract_short_answer(s)
        norm = normalize_answer(short)
        assigned = None
        for ci, (rep_short, rep_norm) in enumerate(reps):
            if norm == rep_norm:
                assigned = ci
                break
            if norm == "" or rep_norm == "":
                continue
            calls += 1
            if not entails(question, rep_short, short):
                continue
            calls += 1
            if entails(question, short, rep_short):
                assigned = ci
                break
        if assigned is None:
            reps.append((short, norm))
            assigned = len(reps) - 1
        ids.append(assigned)
    return ids, calls


def semantic_entropy_from_ids(cluster_ids, k=None):
    """Discrete semantic entropy (nats) from cluster ids of the first k samples (all if k is None).
    Because clustering is greedy and sequential, the first k ids equal the clustering of the first k samples alone."""
    ids = list(cluster_ids if k is None else cluster_ids[:k])
    n = len(ids)
    if n == 0:
        return float("nan")
    counts = Counter(ids)
    return float(-sum((c / n) * math.log(c / n) for c in counts.values()))


class NLIEntailment:
    """entails(question, a, b): True if the NLI model's argmax label for (premise='q a', hypothesis='q b') is entailment."""

    def __init__(self, model_name, device, max_length=256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device, self.max_length = device, max_length
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        hits = [i for i, lab in id2label.items() if "entail" in lab.lower()]
        if len(hits) != 1:
            raise RuntimeError(f"cannot identify the entailment label in {id2label}")
        self.entail_idx, self.id2label = hits[0], id2label
        self.cache = {}
        self.n_model_calls = 0

    def __call__(self, question, a, b):
        key = (question, a, b)
        if key in self.cache:
            return self.cache[key]
        enc = self.tok(f"{question} {a}", f"{question} {b}", return_tensors="pt", truncation=True,
                       max_length=self.max_length).to(self.device)
        with self.torch.no_grad():
            pred = int(self.model(**enc).logits[0].argmax())
        self.n_model_calls += 1
        self.cache[key] = (pred == self.entail_idx)
        return self.cache[key]
