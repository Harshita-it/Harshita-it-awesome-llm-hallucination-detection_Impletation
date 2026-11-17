"""Cycle 4: merge raw generations + semantic-entropy results into the per-question table.
The public table contains NO question text, NO gold answers and NO model answer strings (a correct answer equals the gold
answer, and TriviaQA's terms are unspecified); those stay in the local, git-ignored results/raw/ files."""
import numpy as np
import pandas as pd

from .scoring import aggregate_scores
from .semantic_entropy import semantic_entropy_from_ids
from .textnorm import is_correct, is_correct_containment

NAME = {"mean_nll": "meanNLL", "sum_nll": "sumNLL", "max_nll": "maxNLL", "mean_entropy": "meanH", "max_entropy": "maxH"}
TOKEN_VARIANTS = [f"{ts}_{a}" for ts in ("all", "span") for a in NAME.values()]


def row_from_records(gen, sem=None):
    """gen: one raw generation record (see generation.generate_record plus question_id/pool_index/split/is_pilot and
    normalized_aliases). sem: optional semantic-entropy record."""
    lp, ent, eos, mask = gen["token_logprob"], gen["token_entropy"], gen["is_eos"], gen["span_mask"]
    keep_all = [i for i, e in enumerate(eos) if not e]
    keep_span = [i for i in keep_all if mask[i]]
    fallback = len(keep_span) == 0 and len(keep_all) > 0
    if fallback:
        keep_span = keep_all           # answer text not located: use all tokens and flag it
    row = {"question_id": gen["question_id"], "pool_index": gen["pool_index"], "split": gen["split"],
           "is_pilot": bool(gen["is_pilot"]), "n_tokens": len(keep_all),
           "truncated": bool(gen["truncated"]), "span_fallback": bool(fallback),
           "argmax_match_rate": gen.get("argmax_match_rate", np.nan)}
    for ts, keep in (("all", keep_all), ("span", keep_span)):
        agg = aggregate_scores([lp[i] for i in keep], [ent[i] for i in keep])
        for k, nm in NAME.items():
            row[f"{ts}_{nm}"] = agg[k]
    aliases = gen["normalized_aliases"]
    row["y_wrong_exact"] = int(not is_correct(gen["greedy_text"], aliases))
    row["y_wrong_contain"] = int(not is_correct_containment(gen["greedy_text"], aliases))
    row["seconds_greedy"] = gen["seconds_greedy"]
    row["seconds_samples"] = gen["seconds_samples"]
    if sem is not None:
        ids = sem["cluster_ids"]
        row.update({"n_clusters": len(set(ids)), "SE10": semantic_entropy_from_ids(ids),
                    "SE5": semantic_entropy_from_ids(ids, k=5), "nli_calls": sem["nli_calls"],
                    "seconds_nli": sem["seconds_nli"]})
    else:
        row.update({"n_clusters": np.nan, "SE10": np.nan, "SE5": np.nan, "nli_calls": np.nan, "seconds_nli": np.nan})
    return row


def build_table(gens, sems):
    """gens: list of raw generation records; sems: list of semantic records. Returns DataFrame sorted by pool_index."""
    by_id = {s["question_id"]: s for s in sems}
    rows = [row_from_records(g, by_id.get(g["question_id"])) for g in gens]
    return pd.DataFrame(rows).sort_values("pool_index").reset_index(drop=True)
