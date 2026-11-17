"""Hand-check helpers: blind label check (protocol Section 6, step 5) and error-analysis sampling."""
import random

YES = {"y", "yes", "1", "true", "correct", "c"}
NO = {"n", "no", "0", "false", "wrong", "w"}
ERROR_CATEGORIES = ["genuine_model_error", "missing_alias_or_label_noise", "nli_over_fragmentation", "truncation", "other"]


def parse_human_label(x):
    """Return True (answer is correct), False (answer is wrong) or None (blank/unclear)."""
    if x is None:
        return None
    t = str(x).strip().lower()
    if t in YES:
        return True
    if t in NO:
        return False
    return None


def select_for_labelcheck(records, n=30, seed=42):
    """Deterministic sample of n pilot records (sorted by pool_index first). Returns the list sorted by pool_index."""
    recs = sorted(records, key=lambda r: r["pool_index"])
    if len(recs) <= n:
        return recs
    return sorted(random.Random(seed).sample(recs, n), key=lambda r: r["pool_index"])


def agreement(auto_correct, human_correct):
    """auto_correct, human_correct: parallel lists of bool/None. Unclear human labels are excluded and counted."""
    pairs = [(a, h) for a, h in zip(auto_correct, human_correct) if h is not None]
    n = len(pairs)
    agree = sum(1 for a, h in pairs if a == h)
    return {"n_compared": n, "n_unclear_excluded": len(auto_correct) - n, "agreement": (agree / n) if n else float("nan"),
            "auto_correct_human_wrong": sum(1 for a, h in pairs if a and not h),
            "auto_wrong_human_correct": sum(1 for a, h in pairs if (not a) and h)}


def select_error_analysis(table_rows, n_each=30):
    """table_rows: list of dicts with question_id, SE10, y_wrong_exact (test rows). Returns (confident_errors, uncertain_correct):
    the n wrong answers with the LOWEST SE10 and the n correct answers with the HIGHEST SE10 (ties broken by question_id)."""
    wrong = sorted([r for r in table_rows if r["y_wrong_exact"] == 1], key=lambda r: (r["SE10"], r["question_id"]))[:n_each]
    right = sorted([r for r in table_rows if r["y_wrong_exact"] == 0], key=lambda r: (-r["SE10"], r["question_id"]))[:n_each]
    return wrong, right


# ---------------------------------------------------------------- interactive helpers (used from the Colab notebook)
def ask_choice(prompt, choices, input_fn=input, show=print):
    """Ask until the answer (lower-cased, stripped) is one of `choices`."""
    while True:
        ans = str(input_fn(prompt)).strip().lower()
        if ans in choices:
            return ans
        show(f"  please answer one of: {sorted(choices)}")


def interactive_labelcheck(rows, input_fn=input, show=print):
    """Blind label check: shows ONLY question, model answer, gold answer and aliases (never an uncertainty score).
    rows: list of dicts with question, model_answer, gold_value, gold_aliases, human_correct. Rows already filled are skipped.
    Answers: y / n / ? (unclear, stored as 'unclear') / q (stop and keep progress). Returns the rows (modified in place)."""
    for i, r in enumerate(rows, 1):
        if str(r.get("human_correct", "")).strip():
            continue
        show(f"\n[{i}/{len(rows)}] QUESTION: {r['question']}\n    model answer : {r['model_answer']}\n"
             f"    gold answer  : {r['gold_value']}   (aliases: {r['gold_aliases']})")
        a = ask_choice("Is the model answer CORRECT? y / n / ? (unclear) / q (stop and save): ", {"y", "n", "?", "q"}, input_fn, show)
        if a == "q":
            break
        r["human_correct"] = {"y": "yes", "n": "no", "?": "unclear"}[a]
    return rows


def interactive_error_categories(rows, input_fn=input, show=print):
    """Error analysis: assign one category per row (column `category`). Rows already categorized are skipped. q stops and keeps progress."""
    cats = {str(i + 1): c for i, c in enumerate(ERROR_CATEGORIES)}
    legend = "  ".join(f"{k}={v}" for k, v in cats.items())
    for i, r in enumerate(rows, 1):
        if str(r.get("category", "")).strip():
            continue
        show(f"\n[{i}/{len(rows)}] ({r['group']}) QUESTION: {r['question']}\n    model answer : {r['model_answer']}\n"
             f"    gold answer  : {r['gold_value']} | clusters among 10 samples: {r.get('n_clusters', '?')}")
        a = ask_choice(f"Category? {legend} | q=stop and save: ", set(cats) | {"q"}, input_fn, show)
        if a == "q":
            break
        r["category"] = cats[a]
    return rows
