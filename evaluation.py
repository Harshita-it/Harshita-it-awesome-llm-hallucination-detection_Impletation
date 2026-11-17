"""Cycle 6/7: evaluation exactly as written in experiment_protocol.md (Sections 9 to 12) + figures.

mode="pilot": uses ONLY the pilot rows (calibration part); no test row is read; no calibration/cascade analysis.
mode="final": calibration part fits Platt models and escalation thresholds; the test part is evaluated once.
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cascade as C
from . import metrics as M
from .table import TOKEN_VARIANTS

PRIMARY = "all_meanNLL"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def run_analysis(df, cfg, mode, out_dir, n_boot=None, make_figs=True, ignore_balance=False, provenance=None):
    assert mode in ("pilot", "final")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bcfg = cfg["metrics"]["bootstrap"]
    B, seed = int(n_boot or bcfg["n"]), int(bcfg["seed"])
    ci_level = float(bcfg["ci"])
    rules = cfg["decision_rules"]
    flags = []
    need = [PRIMARY, "SE10", "n_tokens", "y_wrong_exact"] + TOKEN_VARIANTS

    if mode == "pilot":
        ev = df[df["is_pilot"].astype(bool)].copy()
        cal = None
    else:
        ev = df[df["split"] == "test"].copy()
        cal = df[df["split"] == "calibration"].copy()
    n0 = len(ev)
    ev = ev.dropna(subset=need).reset_index(drop=True)
    if n0 != len(ev):
        flags.append(f"{n0 - len(ev)} of {n0} evaluation rows dropped for missing scores (listwise, same rows for all methods)")
    if cal is not None:
        c0 = len(cal)
        cal = cal.dropna(subset=need).reset_index(drop=True)
        if c0 != len(cal):
            flags.append(f"{c0 - len(cal)} of {c0} calibration rows dropped for missing scores")
    if len(ev) < 10:
        raise RuntimeError(f"only {len(ev)} usable evaluation rows")

    y = ev["y_wrong_exact"].to_numpy().astype(int)
    n, n_wrong = len(y), int(y.sum())
    n_correct = n - n_wrong
    if min(n_wrong, n_correct) == 0:
        raise RuntimeError("evaluation rows contain only one class; AUROC undefined")
    min_cls = int(cfg["dataset"]["min_per_class_in_test"])
    balance_ok = min(n_wrong, n_correct) >= min_cls
    if mode == "final" and not balance_ok:
        msg = f"class balance violated: {n_correct} correct / {n_wrong} wrong in test (need >= {min_cls} each)"
        if not ignore_balance:
            raise RuntimeError(msg + ". Protocol Section 4 step 7: enlarge the pool (class counts only; do not look at scores).")
        flags.append(msg + " (ignored by --ignore-balance)")

    idx = M.bootstrap_indices(n, B, seed)
    rand = M.random_scores(n, seed)
    S = {"token": ev[PRIMARY].to_numpy(float), "se": ev["SE10"].to_numpy(float),
         "length": ev["n_tokens"].to_numpy(float), "random": rand}
    LABEL = {"token": "Mean NLL, all tokens (primary)", "se": "Semantic entropy (N=10)",
             "length": "Answer length only (control)", "random": "Random score (control)"}

    # ---------------- Table 2: main comparison
    platt, p_ev = {}, {}
    if cal is not None:
        cal_scores = {"token": cal[PRIMARY].to_numpy(float), "se": cal["SE10"].to_numpy(float),
                      "length": cal["n_tokens"].to_numpy(float), "random": M.random_scores(len(cal), seed + 1)}
        yc = cal["y_wrong_exact"].to_numpy().astype(int)
        for k in S:
            platt[k] = M.PlattScaler().fit(cal_scores[k], yc)
            p_ev[k] = platt[k].predict(S[k])
    boots = {k: M.boot_stat(M.auroc, y, S[k], idx) for k in S}
    rows2 = []
    for k in S:
        lo, hi = M.percentile_ci(boots[k], ci_level)
        ap = M.boot_stat(M.auprc, y, S[k], idx)
        alo, ahi = M.percentile_ci(ap, ci_level)
        rows2.append({"score": LABEL[k], "key": k, "auroc": M.auroc(y, S[k]), "auroc_lo": lo, "auroc_hi": hi,
                      "auprc": M.auprc(y, S[k]), "auprc_lo": alo, "auprc_hi": ahi,
                      "ece": M.ece(y, p_ev[k]) if k in p_ev else np.nan,
                      "brier": M.brier(y, p_ev[k]) if k in p_ev else np.nan,
                      "platt_slope": platt[k].slope if k in platt else np.nan,
                      "n": n, "n_wrong": n_wrong, "prevalence_wrong": n_wrong / n})
    t2 = pd.DataFrame(rows2)
    t2.to_csv(out_dir / "table2.csv", index=False)

    # ---------------- RQ1
    d_boot = boots["se"] - boots["token"]
    d_lo, d_hi = M.percentile_ci(d_boot, ci_level)
    tok_lo = float(t2.loc[t2.key == "token", "auroc_lo"].iloc[0])
    rq1 = {"token_auroc": M.auroc(y, S["token"]), "token_auroc_ci": [tok_lo, float(t2.loc[t2.key == "token", "auroc_hi"].iloc[0])],
           "token_is_predictive": bool(tok_lo > rules["rq1"]["predictive_if_ci_lower_above"]),
           "se_auroc": M.auroc(y, S["se"]), "diff_se_minus_token": M.auroc(y, S["se"]) - M.auroc(y, S["token"]),
           "diff_ci": [d_lo, d_hi],
           "H1": "supported" if d_lo > 0 else "not supported (CI of the difference includes 0 or favours the token score)"}

    # ---------------- RQ2: aggregation ablation, contrasts, length
    vboots = {v: M.boot_stat(M.auroc, y, ev[v].to_numpy(float), idx) for v in TOKEN_VARIANTS}
    rows3 = []
    for v in TOKEN_VARIANTS:
        lo, hi = M.percentile_ci(vboots[v], ci_level)
        rows3.append({"kind": "variant", "name": v, "auroc": M.auroc(y, ev[v].to_numpy(float)), "lo": lo, "hi": hi,
                      "matters": np.nan})
    c_level = float(rules["rq2"]["ci"])
    min_diff = float(rules["rq2"]["min_abs_diff"])
    contrasts = [("C1_span_vs_all_meanNLL", "span_meanNLL", "all_meanNLL"), ("C2_maxNLL_vs_meanNLL", "all_maxNLL", "all_meanNLL"),
                 ("C3_meanH_vs_meanNLL", "all_meanH", "all_meanNLL")]
    rq2 = {"contrasts": {}}
    for name, a, b in contrasts:
        pt = M.auroc(y, ev[a].to_numpy(float)) - M.auroc(y, ev[b].to_numpy(float))
        lo, hi = M.percentile_ci(vboots[a] - vboots[b], c_level)
        matters = bool((lo > 0 or hi < 0) and abs(pt) >= min_diff)
        rows3.append({"kind": "contrast", "name": name, "auroc": pt, "lo": lo, "hi": hi, "matters": matters})
        rq2["contrasts"][name] = {"difference": pt, "ci": [lo, hi], "ci_level": c_level, "matters": matters}
    t3 = pd.DataFrame(rows3)
    t3.to_csv(out_dir / "table3.csv", index=False)
    len_lo = float(t2.loc[t2.key == "length", "auroc_lo"].iloc[0])
    rho, pval = M.spearman(S["token"], S["length"])
    nt = S["length"]
    edges = np.unique(np.quantile(nt, [1 / 3, 2 / 3]))
    if len(edges) < 2:
        edges = np.unique(np.quantile(nt, [0.5]))
    strata = []
    if len(edges) >= 1 and len(np.unique(nt)) > 1:
        g = np.digitize(nt, edges)
        for gi in np.unique(g):
            m = g == gi
            strata.append({"stratum": int(gi), "n": int(m.sum()), "n_wrong": int(y[m].sum()),
                           "auroc_token": M.auroc(y[m], S["token"][m]), "auroc_se": M.auroc(y[m], S["se"][m]),
                           "token_count_range": [float(nt[m].min()), float(nt[m].max())]})
    rq2.update({"length_only_auroc": float(t2.loc[t2.key == "length", "auroc"].iloc[0]), "length_auroc_ci_lower": len_lo,
                "length_confound_present": bool(len_lo > 0.5), "spearman_token_score_vs_length": rho, "spearman_p": pval,
                "length_strata": strata})

    # ---------------- RQ3: cascade (final only)
    rq3, t4 = None, None
    if cal is not None:
        q_primary = float(cfg["cascade"]["primary_quantile"])
        quants = sorted(set(float(q) for q in cfg["cascade"]["curve_quantiles"]) | {q_primary})
        s_cal, se_cal = cal[PRIMARY].to_numpy(float), cal["SE10"].to_numpy(float)
        nli = ev["nli_calls"].fillna(0).to_numpy(float)
        rows4, cboots = [], {}
        for q in quants:
            risk, esc = C.cascade_risk(q, s_cal, S["token"], S["se"], platt["token"], platt["se"])
            bs = M.boot_stat(M.auroc, y, risk, idx)
            cboots[q] = bs
            lo, hi = M.percentile_ci(bs, ci_level)
            rr = C.rank_rule_risk(esc, S["token"], S["se"])
            cost = C.cost_summary(esc, nli, ev["seconds_greedy"].to_numpy(float), ev["seconds_samples"].to_numpy(float),
                                  ev["seconds_nli"].fillna(0).to_numpy(float), cfg["generation"]["sampling"]["num_samples"])
            rows4.append({"quantile_q": q, "role": "token-only" if q >= 1 else ("SE-for-all" if q <= 0 else ("primary" if q == q_primary else "curve")),
                          "auroc": M.auroc(y, risk), "auroc_lo": lo, "auroc_hi": hi, "auroc_rank_rule": M.auroc(y, rr), **cost})
        t4 = pd.DataFrame(rows4)
        t4.to_csv(out_dir / "table4.csv", index=False)
        prim = t4.loc[t4.quantile_q == q_primary].iloc[0]
        D = M.auroc(y, S["se"]) - float(prim["auroc"])
        D_boot = boots["se"] - cboots[q_primary]
        D_lo, D_hi = M.percentile_ci(D_boot, ci_level)
        margin, upper = float(rules["rq3"]["margin"]), float(rules["rq3"]["upper_ci_limit"])
        outcome = "not supported" if D >= margin else ("supported" if D_hi < upper else "inconclusive")
        rq3 = {"primary_quantile": q_primary, "realized_escalated_fraction": float(prim["escalated_fraction"]),
               "auroc_cascade": float(prim["auroc"]), "auroc_se_raw": M.auroc(y, S["se"]), "D": D, "D_ci": [D_lo, D_hi],
               "margin": margin, "upper_ci_limit": upper, "H3": outcome,
               "extra_generations_per_question_cascade": float(prim["extra_generations_per_question"]),
               "extra_generations_per_question_se_for_all": float(cfg["generation"]["sampling"]["num_samples"]),
               "platt_slope_token": platt["token"].slope, "platt_slope_se": platt["se"].slope}
        for k in ("token", "se"):
            if platt[k].slope <= 0:
                flags.append(f"Platt slope for {k} is {platt[k].slope:.3f} <= 0: the calibrated ranking is inverted relative to the raw score")
        if rq1["token_auroc"] < 0.5 + 1e-9 or not rq1["token_is_predictive"]:
            flags.append("RQ1: token score is not predictive by the pre-specified rule; RQ3 is expected to fail and must be reported as such")

    # ---------------- Sensitivity (Table 6)
    sens = []
    se5 = ev["SE5"].to_numpy(float)
    if not np.isnan(se5).any():
        bs = M.boot_stat(M.auroc, y, se5, idx)
        lo, hi = M.percentile_ci(bs, ci_level)
        sens.append({"analysis": "SE from first 5 of 10 samples", "auroc": M.auroc(y, se5), "lo": lo, "hi": hi})
    if "y_wrong_contain" in ev and not ev["y_wrong_contain"].isna().any():
        yc2 = ev["y_wrong_contain"].to_numpy().astype(int)
        for k, lab in (("token", "Mean NLL"), ("se", "Semantic entropy")):
            bs = M.boot_stat(M.auroc, yc2, S[k], idx)
            lo, hi = M.percentile_ci(bs, ci_level)
            sens.append({"analysis": f"{lab}, containment label", "auroc": M.auroc(yc2, S[k]), "lo": lo, "hi": hi})
    pd.DataFrame(sens).to_csv(out_dir / "table6.csv", index=False)

    # ---------------- results.csv (long format) and summary.json
    long_rows = []
    for r in rows2:
        for m in ("auroc", "auprc", "ece", "brier"):
            long_rows.append({"mode": mode, "analysis": "main", "name": r["score"], "metric": m, "value": r[m],
                              "ci_lo": r.get(m + "_lo", np.nan), "ci_hi": r.get(m + "_hi", np.nan), "n": n})
    for r in rows3:
        long_rows.append({"mode": mode, "analysis": "ablation_" + r["kind"], "name": r["name"], "metric": "auroc_or_difference",
                          "value": r["auroc"], "ci_lo": r["lo"], "ci_hi": r["hi"], "n": n})
    if t4 is not None:
        for _, r in t4.iterrows():
            long_rows.append({"mode": mode, "analysis": "cascade", "name": f"q={r.quantile_q}", "metric": "auroc",
                              "value": r.auroc, "ci_lo": r.auroc_lo, "ci_hi": r.auroc_hi, "n": n})
    for r in sens:
        long_rows.append({"mode": mode, "analysis": "sensitivity", "name": r["analysis"], "metric": "auroc",
                          "value": r["auroc"], "ci_lo": r["lo"], "ci_hi": r["hi"], "n": n})
    res_df = pd.DataFrame(long_rows)
    for k, v in (provenance or {}).items():          # traceability: every row carries where it came from
        res_df[k] = v
    res_df.to_csv(out_dir / "results.csv", index=False)

    summary = {"mode": mode, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": _git_commit(),
               "n_eval": n, "n_wrong": n_wrong, "n_correct": n_correct, "class_balance_ok": balance_ok,
               "n_calibration": None if cal is None else len(cal), "bootstrap": {"n": B, "seed": seed, "ci_level": ci_level},
               "rq1": rq1, "rq2": rq2, "rq3": rq3, "flags": flags, "provenance": provenance,
               "note": "pilot mode is exploratory and uses calibration-part questions only" if mode == "pilot" else "final mode"}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    if make_figs:
        _figures(out_dir, y, S, ev, t3, t4, p_ev, idx, seed)
    return summary


def _figures(out_dir, y, S, ev, t3, t4, p_ev, idx, seed, n_fig_boot=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                                "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                                "figure.dpi": 100, "savefig.dpi": 300})
    fd = Path(out_dir) / "figs"
    fd.mkdir(exist_ok=True)

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(fd / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    labels = {"token": "Mean NLL (primary)", "se": "Semantic entropy", "length": "Length only", "random": "Random"}
    # Fig 2: ROC curves with bootstrap band (n_fig_boot resamples)
    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for k, sc in S.items():
        fpr, tpr, _ = roc_curve(y, sc)
        ax.plot(fpr, tpr, label=f"{labels[k]} (AUROC {M.auroc(y, sc):.3f})")
        tprs = []
        for i in idx[:n_fig_boot]:
            if y[i].min() == y[i].max():
                continue
            f_, t_, _ = roc_curve(y[i], sc[i])
            tprs.append(np.interp(grid, f_, t_))
        if tprs:
            ax.fill_between(grid, np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0), alpha=0.12)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (wrong answers)")
    ax.legend(fontsize=7, loc="lower right"); save(fig, "fig2_roc")
    # Fig 3: AUROC per aggregator
    v = t3[t3.kind == "variant"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ypos = np.arange(len(v))
    ax.errorbar(v["auroc"], ypos, xerr=[v["auroc"] - v["lo"], v["hi"] - v["auroc"]], fmt="o", capsize=3)
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    pretty = {"meanNLL": "mean NLL", "sumNLL": "sum NLL", "maxNLL": "max NLL", "meanH": "mean entropy", "maxH": "max entropy"}

    def pn(name):
        ts, k = name.split("_", 1)
        return f"{pretty[k]} ({'all tokens' if ts == 'all' else 'answer span'})"
    ax.set_yticks(ypos); ax.set_yticklabels([pn(x) for x in v["name"]], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("AUROC (95% CI)"); save(fig, "fig3_aggregators")
    # Fig 4: cascade curve (final only)
    if t4 is not None:
        d = t4.sort_values("escalated_fraction")
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        l1, = ax.plot(d["escalated_fraction"], d["auroc"], "o-", color="tab:blue", label="Cascade AUROC (95% CI band)")
        ax.fill_between(d["escalated_fraction"], d["auroc_lo"], d["auroc_hi"], alpha=0.15)
        pr = d[d.role == "primary"]
        if len(pr):
            ax.axvline(float(pr["escalated_fraction"].iloc[0]), color="grey", ls="--", lw=0.8)
        ax.set_xlabel("Fraction of questions escalated to semantic entropy"); ax.set_ylabel("AUROC")
        ax2 = ax.twinx()
        l2, = ax2.plot(d["escalated_fraction"], d["seconds_per_question"], "s--", color="tab:red", label="Measured seconds per question")
        ax2.set_ylabel("Measured seconds per question", color="tab:red"); ax2.grid(False)
        ax.set_ylabel("AUROC", color="tab:blue")
        ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left", fontsize=7)
        save(fig, "fig4_cascade")
    # Fig 5: reliability diagram (final only)
    if p_ev:
        fig, ax = plt.subplots(figsize=(4.2, 4))
        for k in ("token", "se"):
            rt = [r for r in M.reliability_table(y, p_ev[k]) if r["n"] > 0]
            ax.plot([r["mean_pred"] for r in rt], [r["frac_wrong"] for r in rt], "o-", label=labels[k])
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("Predicted probability of a wrong answer"); ax.set_ylabel("Observed fraction wrong")
        ax.legend(fontsize=7); save(fig, "fig5_reliability")
