"""Reproducibility comparison of two generation runs on the same questions."""


def compare_generation_runs(a, b, lp_tol=0.05):
    A = {r["question_id"]: r for r in a}
    B = {r["question_id"]: r for r in b}
    common = sorted(set(A) & set(B))
    same_greedy = sum(A[q]["greedy_text"] == B[q]["greedy_text"] for q in common)
    same_samples = sum(A[q]["samples"] == B[q]["samples"] for q in common)
    diffs = [abs(x - y) for q in common if len(A[q]["token_logprob"]) == len(B[q]["token_logprob"])
             for x, y in zip(A[q]["token_logprob"], B[q]["token_logprob"])]
    max_diff = max(diffs) if diffs else float("nan")
    n = len(common)
    return {"n_common": n, "greedy_identical": same_greedy, "samples_identical": same_samples,
            "max_abs_token_logprob_diff": max_diff,
            "greedy_ok": n > 0 and same_greedy == n, "logprob_ok": bool(diffs) and max_diff <= lp_tol,
            "samples_ok": n > 0 and same_samples == n}
