"""Pre-evaluation audit (Prompt 13): runs BEFORE any metric is computed.
Checks ID/label alignment, missing values, partitions, metric direction/ranges, leakage and comparator consistency.
Severity: ERROR = stop (evaluation refuses to run) | WARN = report and explain | INFO | SKIPPED = an input was not available,
so the check did NOT run (never silently counted as passed)."""
import math
import time

import numpy as np
import pandas as pd

from . import metrics as M
from .data import ids_hash
from .semantic_entropy import semantic_entropy_from_ids
from .table import TOKEN_VARIANTS
from .textnorm import is_correct, normalize_answer

PRIMARY = "all_meanNLL"
REQUIRED = (["question_id", "pool_index", "split", "is_pilot", "y_wrong_exact", "y_wrong_contain", "n_tokens", "truncated",
             "SE10", "SE5", "n_clusters", "nli_calls", "seconds_greedy", "seconds_samples", "seconds_nli"] + TOKEN_VARIANTS)
SCORE_COLS = [PRIMARY, "SE10", "n_tokens"] + TOKEN_VARIANTS
TOL = 1e-9
MISSING_ERROR_FRACTION = 0.05          # provisional threshold: more than 5% missing in an evaluated split is an error


class _F:
    def __init__(self):
        self.items = []

    def add(self, check, sev, msg, **details):
        self.items.append({"check": check, "severity": sev, "message": msg, "details": details})


def _parse_time(s):
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))


def run_audit(df, cfg, mode, ids=None, manifest=None, gens=None, sems=None, metas=None, freeze=None, strict_freeze=True):
    """df: per_question table. ids: subset ids DataFrame. manifest: dict. gens/sems/metas: lists of raw records.
    freeze: dict(tag_exists=bool, record=dict|None, code_matches_record=bool|None) (final mode).
    strict_freeze=False downgrades freeze-related errors to warnings (only for --skip-tag-check; the output says so)."""
    F = _F()
    assert mode in ("pilot", "final")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        F.add("schema", "ERROR", f"missing columns: {missing}")
        return _report(F, mode, len(df))
    df = df.copy()
    F.add("schema", "INFO", f"{len(df)} rows, all {len(REQUIRED)} required columns present")

    # ---- identifiers and partitions
    if df["question_id"].duplicated().any():
        F.add("ids", "ERROR", f"{int(df['question_id'].duplicated().sum())} duplicated question_id")
    if df["pool_index"].duplicated().any() or df["pool_index"].isna().any():
        F.add("ids", "ERROR", "pool_index is missing or duplicated")
    bad_split = sorted(set(df["split"]) - {"calibration", "test"})
    if bad_split:
        F.add("partitions", "ERROR", f"unknown split values: {bad_split}")
    pil = df["is_pilot"].astype(bool)
    if (pil & (df["split"] != "calibration")).any():
        F.add("partitions", "ERROR", "pilot rows outside the calibration part (leakage of test questions into the pilot)")
    cal_ids, test_ids = set(df.loc[df.split == "calibration", "question_id"]), set(df.loc[df.split == "test", "question_id"])
    if cal_ids & test_ids:
        F.add("leakage", "ERROR", f"{len(cal_ids & test_ids)} question_ids appear in BOTH calibration and test")
    else:
        F.add("leakage", "INFO", "no question_id appears in both calibration and test")

    # ---- labels
    for c in ("y_wrong_exact", "y_wrong_contain"):
        if df[c].isna().any() or not set(df[c].dropna().unique()) <= {0, 1}:
            F.add("labels", "ERROR", f"{c} must be 0/1 with no missing values")
    if ((df["y_wrong_contain"] == 1) & (df["y_wrong_exact"] == 0)).any():
        F.add("labels", "ERROR", "exact-match says correct but containment says wrong: impossible, label code inconsistent")

    # ---- scope for completeness and direction (never uses test labels in pilot mode; calibration labels in final mode)
    scope_mask = pil if mode == "pilot" else df["split"].isin(["calibration", "test"])
    scope = df[scope_mask]
    for sp, d in (scope.groupby("split") if len(scope) else []):
        for c in SCORE_COLS:
            frac = float(d[c].isna().mean())
            if frac > MISSING_ERROR_FRACTION:
                F.add("missing", "ERROR", f"{c} missing in {frac:.1%} of {sp} rows", split=sp, column=c)
            elif frac > 0:
                F.add("missing", "WARN", f"{c} missing in {int(d[c].isna().sum())} {sp} rows (dropped listwise at evaluation)", split=sp, column=c)
    if len(scope) and not any(f["check"] == "missing" for f in F.items):
        F.add("missing", "INFO", "no missing scores in the evaluated rows")

    # ---- value ranges and internal consistency (hold by construction if the code is right)
    allc = df
    for pref in ("all", "span"):
        for k in ("meanNLL", "sumNLL", "maxNLL", "meanH", "maxH"):
            col = f"{pref}_{k}"
            if (allc[col].dropna() < -TOL).any():
                F.add("ranges", "ERROR", f"{col} has negative values (NLL and entropy must be >= 0)")
    ok = allc.dropna(subset=["all_meanNLL", "all_maxNLL", "all_sumNLL"])
    if (ok["all_maxNLL"] + TOL < ok["all_meanNLL"]).any():
        F.add("ranges", "ERROR", "max NLL below mean NLL")
    if len(ok) and (np.abs(ok["all_sumNLL"] - ok["all_meanNLL"] * ok["n_tokens"]) > 1e-6 * (1 + ok["all_sumNLL"].abs())).any():
        F.add("ranges", "ERROR", "all_sumNLL != all_meanNLL * n_tokens")
    se = allc.dropna(subset=["SE10"])
    n_s = int(cfg["generation"]["sampling"]["num_samples"])
    if len(se):
        if ((se["SE10"] < -TOL) | (se["SE10"] > math.log(n_s) + 1e-9)).any():
            F.add("ranges", "ERROR", f"SE10 outside [0, ln {n_s}]")
        if ((se["SE5"] < -TOL) | (se["SE5"] > math.log(5) + 1e-9)).any():
            F.add("ranges", "ERROR", "SE5 outside [0, ln 5]")
        if ((se["n_clusters"] < 1) | (se["n_clusters"] > n_s)).any():
            F.add("ranges", "ERROR", f"n_clusters outside [1, {n_s}]")
        if (((se["SE10"].abs() < 1e-12) != (se["n_clusters"] == 1))).any():
            F.add("ranges", "ERROR", "SE10 == 0 must hold exactly when n_clusters == 1")
    if (allc[["seconds_greedy", "seconds_samples"]].dropna() < 0).any().any() or (allc["seconds_nli"].dropna() < 0).any():
        F.add("ranges", "ERROR", "negative timing values")
    if not any(f["check"] == "ranges" and f["severity"] == "ERROR" for f in F.items):
        F.add("ranges", "INFO", "value ranges and internal identities hold")

    # ---- metric direction (higher = more uncertain = predicts WRONG). Uses pilot/calibration labels only.
    dscope = df[pil] if mode == "pilot" else df[df.split == "calibration"]
    dscope = dscope.dropna(subset=SCORE_COLS)
    if len(dscope) >= 20 and dscope["y_wrong_exact"].nunique() == 2:
        y = dscope["y_wrong_exact"].to_numpy()
        flipped, const, vals = [], [], {}
        for c in SCORE_COLS:
            s = dscope[c].to_numpy(float)
            if np.nanstd(s) == 0:
                const.append(c)
                continue
            a = M.auroc(y, s)
            vals[c] = round(a, 4)
            if a < 0.5:
                flipped.append((c, round(a, 4)))
        if [c for c in const if c != "n_tokens"]:
            F.add("direction", "ERROR", f"constant score columns (no information): {[c for c in const if c != 'n_tokens']}")
        if "n_tokens" in const:
            F.add("direction", "WARN", "answer length is constant: the length-only control carries no information")
        if flipped:
            F.add("direction", "WARN", "AUROC below 0.5 on the " + ("pilot" if mode == "pilot" else "calibration") +
                  f" part for {flipped}: orientation may be inverted OR the score is genuinely anti-predictive. Check before evaluating the test part.",
                  columns=[c for c, _ in flipped])
        F.add("direction", "INFO", "orientation convention: every score is oriented so that HIGHER = more uncertain = predicts a WRONG answer; "
              "AUROC (wrong = positive) is computed on that convention", scope_auroc=vals)
    else:
        F.add("direction", "SKIPPED", "not enough labelled rows with both classes in the pilot/calibration scope to check orientation")

    # ---- class balance
    if mode == "pilot":
        if len(dscope) and dscope["y_wrong_exact"].nunique() < 2:
            F.add("balance", "ERROR", "pilot rows contain only one class: AUROC undefined")
    else:
        mincls = int(cfg["dataset"]["min_per_class_in_test"])
        t = df[df.split == "test"]
        nw, nc = int((t["y_wrong_exact"] == 1).sum()), int((t["y_wrong_exact"] == 0).sum())
        if min(nw, nc) < mincls:
            F.add("balance", "ERROR", f"test has {nc} correct / {nw} wrong; need at least {mincls} of each (protocol Section 4)")
        else:
            F.add("balance", "INFO", f"test class counts: {nc} correct / {nw} wrong")
        c = df[df.split == "calibration"]
        if c["y_wrong_exact"].nunique() < 2:
            F.add("balance", "ERROR", "calibration part contains one class only: Platt scaling impossible")

    # ---- alignment with the saved subset ids
    if ids is None:
        F.add("alignment", "SKIPPED", "data/subset_question_ids.csv not available: alignment with the saved subset not checked")
    else:
        if manifest is not None and ids_hash(ids) != manifest.get("ids_sha256"):
            F.add("alignment", "ERROR", "subset ids file does not match the SHA-256 in its manifest")
        extra = set(df["question_id"]) - set(ids["question_id"])
        if extra:
            F.add("alignment", "ERROR", f"{len(extra)} table rows are not in the saved subset")
        m = df.merge(ids, on="question_id", suffixes=("", "_ids"))
        for col in ("split", "pool_index"):
            if (m[col] != m[col + "_ids"]).any():
                F.add("alignment", "ERROR", f"{col} differs between the table and the saved subset for {int((m[col] != m[col + '_ids']).sum())} rows")
        if (m["is_pilot"].astype(bool) != m["is_pilot_ids"].astype(bool)).any():
            F.add("alignment", "ERROR", "is_pilot differs between the table and the saved subset")
        notyet = set(ids["question_id"]) - set(df["question_id"])
        if notyet:
            F.add("alignment", "WARN" if mode == "final" else "INFO",
                  f"{len(notyet)} subset questions have no row in the table (not generated yet)")
        if not any(f["check"] == "alignment" and f["severity"] == "ERROR" for f in F.items):
            F.add("alignment", "INFO", "table rows agree with the saved subset (ids, split, pool_index, pilot flag)")

    # ---- recomputation from raw records (label alignment and score alignment)
    if gens is None:
        F.add("raw", "SKIPPED", "results/raw/generations.jsonl not available: labels and scores were NOT recomputed from raw outputs")
    else:
        G = {g["question_id"]: g for g in gens}
        bad_lab = bad_tok = bad_nt = 0
        T = df.drop_duplicates("question_id", keep="first").set_index("question_id")   # duplicates are reported above; do not crash on them
        for qid, g in G.items():
            if qid not in T.index:
                continue
            r = T.loc[qid]
            y = int(not is_correct(g["greedy_text"], g["normalized_aliases"]))
            bad_lab += int(pd.isna(r["y_wrong_exact"]) or y != int(r["y_wrong_exact"]))
            keep = [i for i, e in enumerate(g["is_eos"]) if not e]
            bad_nt += int(pd.isna(r["n_tokens"]) or len(keep) != int(r["n_tokens"]))
            if keep and not math.isnan(r["all_meanNLL"]):
                mean_nll = -float(np.mean([g["token_logprob"][i] for i in keep]))
                bad_tok += int(abs(mean_nll - float(r["all_meanNLL"])) > 1e-9)
        for label, k in (("labels recomputed from raw answers", bad_lab), ("n_tokens recomputed", bad_nt), ("mean NLL recomputed", bad_tok)):
            F.add("raw", "ERROR" if k else "INFO", f"{label}: {k} mismatches vs the table" if k else f"{label}: all match the table")
        orphan_g = set(G) - set(df["question_id"])
        if orphan_g:
            F.add("raw", "WARN", f"{len(orphan_g)} raw generations are not in the table (rebuild with scripts/04_build_table.py)")
        n_bad_samples = sum(len(g["samples"]) != n_s for g in G.values())
        if n_bad_samples:
            F.add("comparators", "ERROR", f"{n_bad_samples} records do not have exactly {n_s} sampled answers: semantic entropy not comparable across questions")
        mx = int(cfg["generation"]["greedy"]["max_new_tokens"])
        if any(len(g["token_ids"]) > mx for g in G.values()):
            F.add("comparators", "ERROR", f"a greedy answer exceeds max_new_tokens={mx}: generation settings differ from the protocol")
        if len({g.get("logit_vector_size") for g in G.values()}) > 1:
            F.add("comparators", "WARN", "logit vector size differs between records (different model?)")
        # duplicate question text across splits
        seen = {}
        for g in G.values():
            seen.setdefault(normalize_answer(g["question"]), set()).add(g["split"])
        cross = [t for t, s in seen.items() if len(s) > 1]
        if cross:
            F.add("leakage", "ERROR", f"{len(cross)} question texts appear in more than one split (near-duplicate leakage)")
        else:
            F.add("leakage", "INFO", "no question text appears in more than one split (raw records)")
        # semantic records
        if sems is not None:
            S = {s["question_id"]: s for s in sems}
            orph = set(S) - set(G)
            if orph:
                F.add("raw", "ERROR", f"{len(orph)} semantic-entropy records have no generation record")
            bad_se = 0
            for qid, s in S.items():
                if qid in T.index and not math.isnan(T.loc[qid, "SE10"]):
                    bad_se += int(abs(semantic_entropy_from_ids(s["cluster_ids"]) - float(T.loc[qid, "SE10"])) > 1e-9)
            F.add("raw", "ERROR" if bad_se else "INFO", f"SE10 recomputed from cluster ids: {bad_se} mismatches" if bad_se else "SE10 recomputed from cluster ids: all match")
        # argmax consistency (V1)
        am = [g.get("argmax_match_rate") for g in G.values() if g["split"] != "test" and g.get("argmax_match_rate") is not None]
        if am and float(np.nanmean(am)) < 0.99:
            F.add("comparators", "WARN", f"greedy answers are not the argmax of the teacher-forced scores (mean match {np.nanmean(am):.3f}): scoring and generation disagree")

    # ---- comparator settings across runs
    if metas is None:
        F.add("comparators", "SKIPPED", "results/raw/meta_runs.jsonl not available: environment consistency across runs not checked")
    else:
        gm = [m for m in metas if m.get("stage") == "generate"]
        envs = {(m.get("torch"), m.get("transformers"), m.get("gpu")) for m in gm}
        if len(envs) > 1:
            F.add("comparators", "WARN", f"generation was run in {len(envs)} different environments (torch/transformers/GPU): report this; "
                  "GPU non-determinism may differ between parts", environments=[list(e) for e in envs])
        elif gm:
            F.add("comparators", "INFO", "all generation runs used the same torch/transformers/GPU")
        if len({m.get("protocol_status") for m in gm}) > 1:
            F.add("comparators", "WARN", "generation runs were made under different protocol statuses (draft vs frozen)")

    # ---- protocol order (final mode)
    if mode == "final":
        sev = "ERROR" if strict_freeze else "WARN"
        if freeze is None:
            F.add("protocol", sev, "no freeze information available")
        else:
            if not freeze.get("tag_exists"):
                F.add("protocol", sev, "git tag protocol-v1 not found")
            rec = freeze.get("record")
            if rec is None:
                F.add("protocol", sev, "no freeze record (data/protocol_freeze.json): protocol was never frozen")
            else:
                if freeze.get("code_matches_record") is False:
                    F.add("protocol", sev, "code/config differ from the frozen version (undisclosed protocol change)")
                if metas is not None:
                    tt = [m["time"] for m in metas if m.get("stage") == "generate" and m.get("split") in ("test", "all")]
                    if tt and _parse_time(min(tt)) < _parse_time(rec["time"]):
                        F.add("protocol", sev, f"TEST generation started ({min(tt)}) BEFORE the freeze ({rec['time']})")
                    elif not tt:
                        F.add("protocol", "WARN", "no meta record of test generation: cannot verify that it came after the freeze")
                    else:
                        F.add("protocol", "INFO", "test generation started after the freeze")
            if not any(f["check"] == "protocol" and f["severity"] in ("ERROR", "WARN") for f in F.items):
                F.add("protocol", "INFO", "tag present, code identical to the frozen version")
    else:
        F.add("protocol", "INFO", "pilot mode: freeze checks not applicable (no test rows are read)")
    return _report(F, mode, len(df))


def _report(F, mode, n):
    n_err = sum(f["severity"] == "ERROR" for f in F.items)
    return {"mode": mode, "n_rows": n, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "findings": F.items,
            "n_error": n_err, "n_warn": sum(f["severity"] == "WARN" for f in F.items),
            "n_skipped": sum(f["severity"] == "SKIPPED" for f in F.items), "passed": n_err == 0}


def format_report(rep):
    lines = [f"AUDIT ({rep['mode']}): {rep['n_rows']} rows | {rep['n_error']} ERROR, {rep['n_warn']} WARN, {rep['n_skipped']} SKIPPED"]
    for f in rep["findings"]:
        lines.append(f"  [{f['severity']:<7}] {f['check']:<12} {f['message']}")
    lines.append("AUDIT " + ("PASSED (no ERROR). Read every WARN and SKIPPED line." if rep["passed"] else "FAILED: fix the ERRORs; evaluation will not run."))
    return "\n".join(lines)


def load_audit_inputs(root):
    """Load whatever audit inputs exist on disk (None when a file is missing; SKIPPED checks say so)."""
    import json
    from pathlib import Path

    from .io_utils import read_jsonl
    root = Path(root)

    def jl(p):
        return read_jsonl(p) if p.exists() else None

    ids = pd.read_csv(root / "data" / "subset_question_ids.csv", dtype={"question_id": str}) if (root / "data" / "subset_question_ids.csv").exists() else None
    man = json.load(open(root / "data" / "subset_manifest.json")) if (root / "data" / "subset_manifest.json").exists() else None
    return {"ids": ids, "manifest": man, "gens": jl(root / "results" / "raw" / "generations.jsonl"),
            "sems": jl(root / "results" / "raw" / "semantic.jsonl"), "metas": jl(root / "results" / "raw" / "meta_runs.jsonl")}
