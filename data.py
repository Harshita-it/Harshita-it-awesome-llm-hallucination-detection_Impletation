"""Cycle 1: data loading, validation, deduplication, sampling and splitting (protocol Section 4).

Pipeline:  records -> DataFrame -> dedupe -> validate/drop unlabellable -> sample pool -> split -> save ids only.
Everything is deterministic given the seeds, independent of the order in which rows were loaded.
Only question IDs are saved (not question text), because TriviaQA terms are unspecified (data/README.md).
"""
import hashlib
import json

import numpy as np
import pandas as pd

COLUMNS = ["question_id", "question", "gold_value", "aliases", "normalized_aliases", "question_source"]
IDS_COLUMNS = ["pool_index", "question_id", "split", "is_pilot"]


def rows_from_records(records) -> pd.DataFrame:
    """records: iterable of dicts shaped like a TriviaQA rc.nocontext example
    (keys: question, question_id, question_source, answer{value, aliases, normalized_aliases})."""
    rows = []
    for r in records:
        a = r["answer"]
        rows.append({
            "question_id": str(r["question_id"]),
            "question": str(r["question"]).strip(),
            "gold_value": str(a.get("value", "")).strip(),
            "aliases": list(a.get("aliases", [])),
            "normalized_aliases": list(a.get("normalized_aliases", [])),
            "question_source": str(r.get("question_source", "")),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def dedupe(df: pd.DataFrame):
    """Drop repeated question_ids, then repeated question texts. Deterministic: rows are first sorted by
    (question_id, question_source) so the kept row does not depend on file order.
    Returns (df_clean, stats)."""
    n0 = len(df)
    d = df.sort_values(["question_id", "question_source"], kind="stable").reset_index(drop=True)
    alias_key = d["normalized_aliases"].map(lambda x: tuple(sorted(x)))
    n_conflict = int((d.assign(_k=alias_key).groupby("question_id")["_k"].nunique() > 1).sum())
    n_dup_id_rows = int(d.duplicated("question_id", keep="first").sum())
    d = d.drop_duplicates("question_id", keep="first")
    n_dup_text_rows = int(d.duplicated("question", keep="first").sum())
    d = d.drop_duplicates("question", keep="first").reset_index(drop=True)
    stats = {"rows_before": n0, "dropped_repeated_question_id": n_dup_id_rows,
             "dropped_repeated_question_text": n_dup_text_rows,
             "question_ids_with_conflicting_aliases": n_conflict, "rows_after": len(d)}
    return d, stats


def validate(df: pd.DataFrame) -> dict:
    """Return counts of problems (does not modify df)."""
    return {
        "repeated_question_id": int(df["question_id"].duplicated().sum()),
        "empty_question": int((df["question"].str.len() == 0).sum()),
        "no_normalized_aliases": int((df["normalized_aliases"].map(len) == 0).sum()),
        "empty_gold_value": int((df["gold_value"] == "").sum()),
    }


def drop_unlabellable(df: pd.DataFrame):
    """Rows without any alias cannot be labelled, so they are removed BEFORE sampling (depends on no model output)."""
    keep = df["normalized_aliases"].map(len) > 0
    return df[keep].reset_index(drop=True), int((~keep).sum())


def sample_pool(df: pd.DataFrame, pool_size: int, seed: int) -> pd.DataFrame:
    """Seeded sample of `pool_size` questions. Sorting by question_id first makes the result independent of load order.
    `pool_index` (0..pool_size-1) is the question's position in the pool and defines its sampling seed (1000 + pool_index)."""
    d = df.sort_values("question_id", kind="stable").reset_index(drop=True)
    if pool_size > len(d):
        raise ValueError(f"pool_size {pool_size} > available questions {len(d)}")
    idx = np.random.RandomState(seed).permutation(len(d))[:pool_size]
    pool = d.iloc[idx].reset_index(drop=True)
    pool.insert(0, "pool_index", np.arange(len(pool)))
    return pool


def split_pool(pool: pd.DataFrame, calibration_fraction: float, seed: int, pilot_size: int) -> pd.DataFrame:
    """Split BY QUESTION into calibration/test. The pilot is the first `pilot_size` questions of the calibration part
    (in the permuted order); it never contains test questions."""
    n = len(pool)
    n_cal = int(round(calibration_fraction * n))
    perm = np.random.RandomState(seed).permutation(n)
    cal_idx, test_idx = perm[:n_cal], perm[n_cal:]
    split = np.array(["test"] * n, dtype=object)
    split[cal_idx] = "calibration"
    pilot = np.zeros(n, dtype=bool)
    pilot[cal_idx[:pilot_size]] = True
    out = pool.copy()
    out["split"] = split
    out["is_pilot"] = pilot
    return out


def ids_table(df: pd.DataFrame) -> pd.DataFrame:
    """The only table that is saved: ids and assignments, no question text."""
    return df[IDS_COLUMNS].copy()


def ids_hash(ids_df: pd.DataFrame) -> str:
    s = "\n".join(f"{r.pool_index},{r.question_id},{r.split},{int(r.is_pilot)}" for r in ids_df.itertuples())
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_subset(df_raw: pd.DataFrame, ds_cfg: dict, pool_size=None):
    """Full cycle-1 pipeline on an already loaded DataFrame. Returns (subset_df, report dict)."""
    pool_size = int(pool_size or ds_cfg["pool_size"])
    d, dstats = dedupe(df_raw)
    problems = validate(d)
    d, n_unlabellable = drop_unlabellable(d)
    pool = sample_pool(d, pool_size, ds_cfg["pool_seed"])
    sub = split_pool(pool, ds_cfg["calibration_fraction"], ds_cfg["split_seed"], ds_cfg["pilot_size"])
    ids = ids_table(sub)
    report = {
        "dedupe": dstats, "validation_after_dedupe": problems, "dropped_unlabellable": n_unlabellable,
        "available_after_cleaning": len(d), "pool_size": pool_size,
        "n_calibration": int((sub["split"] == "calibration").sum()), "n_test": int((sub["split"] == "test").sum()),
        "n_pilot": int(sub["is_pilot"].sum()), "ids_sha256": ids_hash(ids),
        "pool_seed": ds_cfg["pool_seed"], "split_seed": ds_cfg["split_seed"],
    }
    return sub, report


def save_subset(sub: pd.DataFrame, report: dict, ids_path, manifest_path, extra: dict = None):
    ids = ids_table(sub)
    ids.to_csv(ids_path, index=False)
    with open(manifest_path, "w") as f:
        json.dump({**report, **(extra or {})}, f, indent=2, default=str)
    return ids


def attach_text(ids: pd.DataFrame, full_clean_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the full subset (with question text and aliases) from the saved ids and a freshly loaded, cleaned dataset."""
    merged = ids.merge(full_clean_df, on="question_id", how="left", validate="one_to_one")
    missing = int(merged["question"].isna().sum())
    if missing:
        raise ValueError(f"{missing} saved question_ids not found in the loaded dataset")
    return merged
