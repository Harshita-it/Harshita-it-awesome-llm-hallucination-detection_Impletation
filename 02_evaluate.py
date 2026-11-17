#!/usr/bin/env python
"""Evaluate a completed per-question CSV. No data/results are fabricated.

Expected columns: question_id, split, y_wrong, token_all_mean_nll, semantic_entropy.
Optional: length_only. Calibration is fit on calibration rows; metrics are reported on test rows.
"""
import argparse
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

REQUIRED = ["question_id", "split", "y_wrong", "token_all_mean_nll", "semantic_entropy"]
SCORES = ["token_all_mean_nll", "semantic_entropy", "length_only"]


def bootstrap_auc(y, score, n=2000, seed=2026):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    vals = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if np.unique(y[ix]).size == 2:
            vals.append(roc_auc_score(y[ix], score[ix]))
    if not vals:
        return {"lower_95": None, "upper_95": None, "valid_resamples": 0}
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"lower_95": float(lo), "upper_95": float(hi), "valid_resamples": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}. Run the experiment pipeline first; no results were generated.")
    df = pd.read_csv(args.input)
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing required columns: {missing_cols}")
    if df["question_id"].isna().any() or df["question_id"].duplicated().any():
        raise SystemExit("question_id must be non-null and unique (one row per question).")
    if df[REQUIRED].isna().any().any():
        raise SystemExit("Missing values found in required columns; resolve them before evaluation.")
    if not set(df["split"].unique()).issubset({"calibration", "test"}):
        raise SystemExit("split must contain only 'calibration' and 'test'.")
    if not {"calibration", "test"}.issubset(set(df["split"])):
        raise SystemExit("Both calibration and test partitions are required.")
    if not set(df["y_wrong"].unique()).issubset({0, 1}):
        raise SystemExit("y_wrong must be binary: 1=wrong, 0=correct.")
    cal = df[df.split == "calibration"].copy()
    test = df[df.split == "test"].copy()
    if cal.y_wrong.nunique() != 2 or test.y_wrong.nunique() != 2:
        raise SystemExit("Both classes (0 and 1) must occur in calibration and test partitions.")
    scores = [c for c in SCORES if c in df.columns]
    for c in scores:
        if not np.isfinite(pd.to_numeric(df[c], errors="coerce")).all():
            raise SystemExit(f"Score column {c} contains non-numeric or non-finite values.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows=[]; predictions={}
    for col in scores:
        xcal=cal[[col]].to_numpy(dtype=float); ycal=cal.y_wrong.to_numpy(dtype=int)
        xtest=test[[col]].to_numpy(dtype=float); ytest=test.y_wrong.to_numpy(dtype=int)
        # Higher raw score is defined as greater uncertainty / greater probability of a wrong answer.
        model=LogisticRegression(solver="lbfgs", random_state=args.seed)
        model.fit(xcal, ycal)
        prob=model.predict_proba(xtest)[:,1]
        auc=float(roc_auc_score(ytest, test[col]))
        apv=float(average_precision_score(ytest, test[col]))
        brier=float(brier_score_loss(ytest, prob))
        frac, mean=calibration_curve(ytest, prob, n_bins=10, strategy="uniform")
        ece=float(np.sum(np.abs(frac-mean)*np.histogram(prob, bins=np.linspace(0,1,11))[0]/len(prob)))
        rows.append({"method":col,"n_test":len(test),"wrong_rate":float(ytest.mean()),"auroc":auc,
                     "auroc_ci_lower_95":bootstrap_auc(ytest,test[col],args.bootstrap,args.seed)["lower_95"],
                     "auroc_ci_upper_95":bootstrap_auc(ytest,test[col],args.bootstrap,args.seed)["upper_95"],
                     "auprc":apv,"ece_10bin_platt":ece,"brier_platt":brier})
        predictions[col]=prob
    out=pd.DataFrame(rows)
    out.to_csv(args.output_dir/"metrics_test.csv",index=False)
    (args.output_dir/"evaluation_manifest.json").write_text(json.dumps({"input":str(args.input),"n_calibration":len(cal),"n_test":len(test),"positive_class":"y_wrong=1","higher_score_means":"more uncertain / more likely wrong","bootstrap_resamples":args.bootstrap,"bootstrap_seed":args.seed,"calibration":"Platt logistic regression fitted on calibration split only","metrics_split":"test"},indent=2))
    for col,prob in predictions.items():
        plt.figure(); plt.hist(prob,bins=20); plt.xlabel("Platt-calibrated P(wrong)"); plt.ylabel("Question count"); plt.title(col); plt.tight_layout(); plt.savefig(args.output_dir/f"{col}_probability_hist.png",dpi=180); plt.close()
    print(out.to_string(index=False))
    print(f"Saved evaluation outputs to {args.output_dir}")

if __name__ == "__main__":
    main()
