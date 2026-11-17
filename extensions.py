"""Pre-registered secondary extension E1: length-adjusted gate score (ONE isolated change; baseline columns are never modified).

Motivation (from my review, limitation L5): token scores can be confounded with answer length. E1 removes the linear
dependence of the primary score on the number of generated tokens, with the line fitted on the CALIBRATION part only.
This is a routine adjustment, not a claimed novelty. It runs only if the length confound is present by the frozen rule."""
import json
from pathlib import Path

import numpy as np

from . import metrics as M
from .evaluation import PRIMARY
from .table import TOKEN_VARIANTS


class LengthAdjuster:
    """score_adj = score - (a + b * n_tokens), with (a, b) from ordinary least squares on calibration data."""

    def fit(self, score, n_tokens):
        x = np.asarray(n_tokens, dtype=float)
        y = np.asarray(score, dtype=float)
        if np.var(x) == 0:
            raise ValueError("n_tokens is constant in the calibration part: no length adjustment is possible")
        self.b = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
        self.a = float(y.mean() - self.b * x.mean())
        return self

    def transform(self, score, n_tokens):
        return np.asarray(score, dtype=float) - (self.a + self.b * np.asarray(n_tokens, dtype=float))


def run_length_adjusted_extension(df, cfg, out_dir, n_boot=None):
    """Compare AUROC of the length-adjusted primary score with the unchanged primary score on the TEST part (paired bootstrap).
    Does not modify df. Writes extension_E1.json. Returns the result dict."""
    ec = cfg["extensions"]["E1_length_adjusted_gate_score"]
    bcfg = cfg["metrics"]["bootstrap"]
    B, seed = int(n_boot or bcfg["n"]), int(bcfg["seed"])
    need = [PRIMARY, "SE10", "n_tokens", "y_wrong_exact"] + TOKEN_VARIANTS      # same rows as the main analysis -> same bootstrap pairing
    cal = df[df["split"] == "calibration"].dropna(subset=need)
    ev = df[df["split"] == "test"].dropna(subset=need).reset_index(drop=True)
    y = ev["y_wrong_exact"].to_numpy().astype(int)
    idx = M.bootstrap_indices(len(y), B, seed)
    out = {"experiment": "E1_length_adjusted_gate_score", "status": ec["status"], "n_eval": int(len(y))}

    len_boot = M.boot_stat(M.auroc, y, ev["n_tokens"].to_numpy(float), idx)
    len_lo, len_hi = M.percentile_ci(len_boot, float(ec["ci"]))
    out.update({"length_only_auroc_ci": [len_lo, len_hi], "trigger_length_confound_present": bool(len_lo > 0.5)})
    if not out["trigger_length_confound_present"]:
        out.update({"ran": False, "reason": "trigger not met: length-only AUROC CI does not exclude 0.5"})
    else:
        try:
            adj = LengthAdjuster().fit(cal[PRIMARY].to_numpy(float), cal["n_tokens"].to_numpy(float))
        except ValueError as e:
            out.update({"ran": False, "reason": str(e)})
        else:
            s0 = ev[PRIMARY].to_numpy(float)
            s1 = adj.transform(s0, ev["n_tokens"].to_numpy(float))
            b0, b1 = M.boot_stat(M.auroc, y, s0, idx), M.boot_stat(M.auroc, y, s1, idx)
            diff = M.auroc(y, s1) - M.auroc(y, s0)
            lo, hi = M.percentile_ci(b1 - b0, float(ec["ci"]))
            verdict = ("improves" if (lo > 0 and diff >= ec["min_abs_diff"]) else
                       "worse" if hi < 0 else "no evidence of improvement")
            out.update({"ran": True, "fit_intercept": adj.a, "fit_slope_per_token": adj.b,
                        "auroc_primary": M.auroc(y, s0), "auroc_adjusted": M.auroc(y, s1), "difference": diff,
                        "difference_ci": [lo, hi], "verdict": verdict,
                        "note": "exploratory secondary analysis; one pre-registered contrast; the primary analysis is unchanged"})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "extension_E1.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out
