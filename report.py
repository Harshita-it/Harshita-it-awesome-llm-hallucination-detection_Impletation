"""Prompt 14: builds the Results-and-Discussion OUTLINE from real output files, with rule-based, non-exaggerating wording.
It computes NO new statistics: every number is copied from results/final/*.csv|json. Mechanisms and alternative explanations are
listed as things to CHECK, never as findings; [AUTHOR] marks what only the author can decide after looking at the evidence."""
import math

import pandas as pd


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def ci(v, lo, hi):
    return f"{f3(v)} (95% CI {f3(lo)} to {f3(hi)})"


def _row(t, col, val):
    r = t[t[col] == val]
    return r.iloc[0] if len(r) else None


def build_outline(summary, t2, t3, t4, t6=None, e1=None, t5=None, handcheck=None):
    S, L = summary, []
    p = S.get("provenance") or {}
    rq1, rq2, rq3 = S["rq1"], S["rq2"], S["rq3"]
    L.append("# Results and Discussion outline (generated from real output files; no new analysis)\n")
    L.append(f"Source: `results/per_question.csv` sha256 {str(p.get('input_sha256'))[:16]}... | commit {p.get('git_commit')} | tag {p.get('protocol_tag')} | "
             f"audit passed: {p.get('audit_passed')} | audit overridden: {p.get('audit_overridden')}\n")
    if not p.get("audit_passed") or p.get("audit_overridden"):
        L.append("> **WARNING: the pre-evaluation audit did not pass or was overridden. Do not interpret these numbers until it passes.**\n")
    for fl in S.get("flags", []):
        L.append(f"> FLAG: {fl}\n")
    n, nw = S["n_eval"], S["n_wrong"]
    L.append(f"## 0. What was evaluated\nTest part: {n} questions, {nw} wrong ({nw / n:.1%}) and {S['n_correct']} correct. Paired bootstrap, "
             f"{S['bootstrap']['n']} resamples, seed {S['bootstrap']['seed']}. Label = normalized exact match with a TriviaQA alias "
             "(not the same thing as 'hallucination'). Metric = AUROC, positive class = wrong answer, higher score = more uncertain.\n")

    # ------------------------------------------------------------------ RQ1
    tok, se = _row(t2, "key", "token"), _row(t2, "key", "se")
    L.append("## 1. RQ1: token score versus semantic entropy\n### Direct observations (copied from the tables)")
    L.append(f"- Mean NLL (primary token score): AUROC {ci(tok.auroc, tok.auroc_lo, tok.auroc_hi)}")
    L.append(f"- Semantic entropy (N=10): AUROC {ci(se.auroc, se.auroc_lo, se.auroc_hi)}")
    L.append(f"- Difference (SE minus token): {ci(rq1['diff_se_minus_token'], rq1['diff_ci'][0], rq1['diff_ci'][1])}")
    for k, nm in (("length", "Length-only control"), ("random", "Random control")):
        r = _row(t2, "key", k)
        L.append(f"- {nm}: AUROC {ci(r.auroc, r.auroc_lo, r.auroc_hi)}")
    lo_d, hi_d = rq1["diff_ci"]
    L.append("\n### Verdict by the pre-registered rule")
    L.append(f"- Token score predictive (lower CI bound above 0.5): **{'yes' if rq1['token_is_predictive'] else 'no'}**")
    L.append(f"- H1 (semantic entropy has the higher AUROC): **{rq1['H1']}**")
    se_lo = float(se.auroc_lo)
    L.append("\n### Wording the evidence allows")
    if se_lo <= 0.5:
        L.append("- The semantic-entropy interval includes 0.5, so semantic entropy is **not shown to be predictive here**. Check the pipeline "
                 "(label audit, cluster viewer, number of one-cluster questions) **before** drawing any scientific conclusion.")
    elif not rq1["token_is_predictive"]:
        L.append(f"- In this setting there is no evidence that the mean-NLL score predicts wrong answers (its interval includes 0.5). "
                 f"Semantic entropy reached AUROC {f3(se.auroc)}.")
        if lo_d > 0:
            L.append("- H1 holds by the pre-registered rule only in this narrow sense: semantic entropy exceeded a score that was **not shown to be above chance**. "
                     "That is not evidence that semantic entropy beats a token score that works.")
    elif lo_d > 0:
        L.append(f"- In this setting semantic entropy had a higher AUROC than mean NLL by {f3(rq1['diff_se_minus_token'])} "
                 f"(95% interval {f3(lo_d)} to {f3(hi_d)}). H1 is supported **for this model, dataset, prompt and label rule**.")
    elif hi_d < 0:
        L.append(f"- Mean NLL had a higher AUROC than semantic entropy by {f3(-rq1['diff_se_minus_token'])}; this **contradicts H1** and the "
                 "review's framing in this setting. Report it as such; do not reinterpret it as support.")
    else:
        L.append("- The interval of the difference includes 0: there is **no evidence of a difference** at this sample size. H1 is not supported.")
    L.append("\n### Not supported by this evidence (do not write)")
    L.append("- 'Semantic entropy is better than token scores' in general, or for other models/datasets/long-form text.")
    if lo_d <= 0 <= hi_d:
        L.append("- 'The two are equivalent/comparable': an interval that includes 0 is **not** evidence of equivalence (no equivalence margin was pre-registered).")
    if not rq1["token_is_predictive"]:
        L.append("- 'Token-level scores do not work': only that this score on this setup was not shown to predict errors.")
    L.append("- Any causal explanation (why one signal is better) as an observed fact.")
    L.append("\n### Alternative explanations to check before writing the discussion")
    L.append("- Label noise (hand-check agreement; error-analysis category 'missing alias / label noise').")
    L.append("- Low diversity of sampled answers (share of questions with one cluster; semantic entropy is then 0 even for confident wrong answers).")
    L.append("- NLI over-fragmentation of short answers (cluster viewer; error-analysis category).")
    L.append("- Length confound (Section 2) and prevalence of wrong answers (AUPRC must be read against it).")
    L.append("- Possible TriviaQA contamination in the model's pre-training (cannot be checked).")
    L.append("\n### Mechanism hypotheses (NOT findings; test or label as conjecture)")
    L.append("- [AUTHOR: state at most one or two conjectures and the evidence that would test each.]\n")

    # ------------------------------------------------------------------ RQ2
    L.append("## 2. RQ2: aggregation and length (exploratory; no hypothesis)\n### Direct observations")
    for _, r in t3[t3.kind == "variant"].iterrows():
        L.append(f"- {r['name']}: AUROC {ci(r.auroc, r.lo, r.hi)}")
    L.append("\n### Pre-specified contrasts (98.3% intervals; only these three are confirmatory)")
    for name, c in rq2["contrasts"].items():
        verdict = ("**matters** (interval excludes 0 and difference at least 0.02)" if c["matters"]
                   else "no evidence that it matters (interval includes 0 or difference below 0.02); this is **not** evidence of equality")
        L.append(f"- {name}: difference {ci(c['difference'], c['ci'][0], c['ci'][1])} -> {verdict}")
    L.append(f"\n- Length-only AUROC {f3(rq2['length_only_auroc'])} (lower CI bound {f3(rq2['length_auroc_ci_lower'])}): length confound "
             f"**{'present' if rq2['length_confound_present'] else 'not shown'}** by the pre-registered rule.")
    L.append(f"- Spearman correlation between the primary score and token count: {f3(rq2['spearman_token_score_vs_length'])} (descriptive; p-value assumes independent questions and is approximate with ties).")
    for st in rq2.get("length_strata", []):
        L.append(f"  - length stratum {st['stratum']} (tokens {st['token_count_range'][0]:.0f} to {st['token_count_range'][1]:.0f}): n={st['n']}, wrong={st['n_wrong']}, "
                 f"AUROC token {f3(st['auroc_token'])}, SE {f3(st['auroc_se'])} (no intervals: small strata)")
    L.append("\n### Not supported\n- Ranking the ten aggregators: all comparisons other than the three contrasts are descriptive.\n"
             "- 'Answer-span scoring helps/does not help' beyond contrast C1 as stated above.")
    L.append("\n### Alternative explanations to check\n- Most answers are only a few tokens long, so answer-span and all-token scores may be nearly identical (share of `span_fallback` rows).\n"
             "- Ties in few-token answers make maximum and minimum aggregators coarse.\n")

    # ------------------------------------------------------------------ RQ3
    prim = _row(t4, "role", "primary")
    tonly, seall = _row(t4, "role", "token-only"), _row(t4, "role", "SE-for-all")
    L.append("## 3. RQ3: token-gated cascade (offline simulation from stored scores)\n### Direct observations")
    L.append(f"- Token only: AUROC {ci(tonly.auroc, tonly.auroc_lo, tonly.auroc_hi)}; extra generations per question 0")
    L.append(f"- Cascade at the primary operating point: AUROC {ci(prim.auroc, prim.auroc_lo, prim.auroc_hi)}; escalated fraction {prim.escalated_fraction:.1%}; "
             f"extra generations per question {prim.extra_generations_per_question:.2f}; NLI calls per question {prim.nli_calls_per_question:.2f} (counted)")
    L.append(f"- Semantic entropy for all questions: AUROC {ci(seall.auroc, seall.auroc_lo, seall.auroc_hi)}; extra generations per question "
             f"{seall.extra_generations_per_question:.2f}")
    L.append(f"- Measured seconds per question: token only {f3(tonly.seconds_per_question)}, cascade {f3(prim.seconds_per_question)}, SE for all {f3(seall.seconds_per_question)} "
             "(one GPU/session; not portable to other hardware)")
    L.append(f"- D = AUROC(SE) minus AUROC(cascade): {ci(rq3['D'], rq3['D_ci'][0], rq3['D_ci'][1])}; margin {rq3['margin']}, upper limit {rq3['upper_ci_limit']}")
    L.append(f"\n### Verdict by the pre-registered rule\n- H3: **{rq3['H3']}**")
    L.append("\n### Wording the evidence allows")
    if rq3["H3"] == "supported":
        L.append(f"- At the primary operating point the cascade kept AUROC within {rq3['margin']} of full semantic entropy (D = {f3(rq3['D'])}; upper bound {f3(rq3['D_ci'][1])}) "
                 f"while escalating {prim.escalated_fraction:.0%} of questions. This holds for the offline simulation in this setting only.")
    elif rq3["H3"] == "inconclusive":
        L.append("- The point estimate of the loss is below the margin but the interval cannot exclude a larger loss: the data are **inconclusive**. "
                 "Do not write that the cascade 'preserves' detection quality.")
    else:
        L.append(f"- The cascade lost at least {rq3['margin']} AUROC relative to full semantic entropy at the primary operating point: H3 is **not supported**. "
                 "Report the compute saved and the quality lost together; do not turn this into a positive claim about efficiency alone.")
    if not rq1["token_is_predictive"]:
        L.append("- RQ1 found no evidence that the gate score predicts errors; a cascade that depends on it was expected to fail (pre-stated).")
    L.append("\n### Not supported\n- 'The cascade reduces latency' as a deployment claim: costs are counted/measured offline on one GPU and do not include batching, caching or engineering overhead.\n"
             "- Any claim about operating points other than the primary one beyond describing the curve (they were not tested as hypotheses).")
    L.append("\n### Alternative explanations to check\n- Platt slope signs (flags above): a negative slope inverts the calibrated ranking of one stage.\n"
             "- Ties in semantic entropy (few distinct values) and in token scores at the threshold.\n- The gate selecting questions by length rather than uncertainty.\n")

    # ------------------------------------------------------------------ E1
    L.append("## 4. E1: length-adjusted gate score (pre-registered, exploratory)")
    if e1 is None:
        L.append("- `extension_E1.json` not found: E1 was not run.\n")
    elif not e1.get("ran"):
        L.append(f"- Not run: {e1.get('reason')}. This is a result of the pre-registered trigger, not a failure.\n")
    else:
        L.append(f"- AUROC adjusted {f3(e1['auroc_adjusted'])} versus primary {f3(e1['auroc_primary'])}; difference {ci(e1['difference'], e1['difference_ci'][0], e1['difference_ci'][1])}; verdict: **{e1['verdict']}** "
                 "(exploratory secondary analysis, one contrast). Do not describe it as an improvement unless the verdict says so.\n")

    # ------------------------------------------------------------------ cost, variation, errors
    L.append("## 5. Variation across runs\nThere is **one stored generation run**. The intervals describe question-sampling variability only; they do not include "
             "the random seed of the sampling, the model, the prompt or the dataset. [AUTHOR: if a rerun comparison exists (scripts/09), report its result here.]\n")
    L.append("## 6. Error and failure modes")
    if t5 is not None and len(t5):
        for _, r in t5.iterrows():
            L.append(f"- {r['group']} / {r['category']}: {int(r['count'])}")
        L.append("Interpretation: [AUTHOR: say which categories dominate and what that implies for the label rule and for the claims above.]\n")
    else:
        L.append("- Error analysis not yet done (`scripts/05_handcheck.py --make-erroranalysis --frozen`). Required before the discussion.\n")
    if handcheck:
        L.append(f"- Label hand-check: agreement {f3(handcheck.get('agreement'))} on {handcheck.get('n_compared')} answers (target {handcheck.get('target')}).")
    if t6 is not None and len(t6):
        L.append("\n### Sensitivity (descriptive)")
        for _, r in t6.iterrows():
            L.append(f"- {r['analysis']}: AUROC {ci(r.auroc, r.lo, r.hi)}")
        L.append("")

    # ------------------------------------------------------------------ limitations
    L.append("## 7. Limitations (state in the paper)")
    for t in ["One small instruction-tuned model, one short-form closed-book dataset, one prompt: no claim beyond this setting (not long-form, multimodal or long-context).",
              "The label is alias matching: a correct answer worded differently counts as wrong; 'wrong answer' is not 'hallucination' or 'confabulation'.",
              "AUROC measures ranking of wrong above correct answers; it does not measure calibration, accuracy-coverage trade-offs or the usefulness of abstention. ECE depends on the binning and on a Platt fit on a small calibration part.",
              "Semantic entropy measures consistency of meaning across samples, not truth: consistently wrong answers get low entropy.",
              "Decision margins (0.02, 0.03, 50%) are my own choices, not derived from the literature.",
              "Single stored run; no estimate of seed-to-seed variance; GPU non-determinism possible.",
              "The cascade is an offline simulation; costs are counted and measured on one GPU.",
              "TriviaQA terms are unspecified by the authors; possible contamination of the model's pre-training cannot be checked; the dataset revision is not pinned.",
              "Hand-check covers 30 pilot answers only."]:
        L.append(f"- {t}")
    L.append("")

    # ------------------------------------------------------------------ follow-ups (max 3)
    wide = float((t2["auroc_hi"] - t2["auroc_lo"]).max()) > 0.10
    if se_lo <= 0.5 or not rq1["token_is_predictive"]:
        first = "A"
    elif rq3["H3"] == "inconclusive" or wide:
        first = "B"
    else:
        first = "C"
    L.append("## 8. Follow-up experiments (at most three; ranked by cost, with likely information gained)")
    L.append("| Rank by cost | Follow-up | Cost | Likely information gained |\n|---|---|---|---|")
    L.append("| 1 (cheapest) | A. Re-analysis of the stored data (cluster-count distribution, label-noise audit, alternative gate scores) **labelled post hoc/exploratory** | no GPU | Separates 'real negative result' from 'pipeline artefact'; weak confirmatory value |")
    L.append("| 2 | B. Second sampling seed and a larger pool of questions | about the cost of the main run (UNKNOWN until timed) | Estimates seed variance and narrows the intervals |")
    L.append("| 3 (most expensive) | C. A second model size or a second closed-book dataset | highest | Tests whether any pattern generalizes |")
    names = {"A": "A (re-analysis of stored data)", "B": "B (second seed, larger pool)", "C": "C (second model or dataset)"}
    why = {"A": "a key result is a negative or null one, so first rule out a pipeline artefact",
           "B": "the cascade verdict is inconclusive or intervals are wide, so more data and a variance estimate carry the most information",
           "C": "the within-setting result is clear, so generalization is the main open question"}
    L.append(f"\nSuggested first, given these results: **{names[first]}**, because {why[first]}.\n")
    return "\n".join(L)
