"""Metrics and statistics (protocol Section 9). Pure numpy/scipy/sklearn; unit-tested."""
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def auroc(y, s):
    """AUROC with positive class y==1 (wrong answer); higher score = predicted positive. Ties get average ranks."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, s):
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, np.asarray(s, dtype=float)))


def bootstrap_indices(n, n_boot, seed):
    """Index matrix [n_boot, n]; reuse the SAME matrix for every score so differences are paired."""
    return np.random.RandomState(seed).randint(0, n, size=(n_boot, n))


def boot_stat(fn, y, s, idx):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    return np.array([fn(y[i], s[i]) for i in idx])


def percentile_ci(values, level=0.95):
    lo = (1 - level) / 2 * 100
    return float(np.nanpercentile(values, lo)), float(np.nanpercentile(values, 100 - lo))


class PlattScaler:
    """One-feature logistic calibration (fit on the CALIBRATION part only)."""

    def fit(self, x, y):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("calibration labels contain only one class")
        self.mu = float(x.mean())
        sd = float(x.std())
        self.sd = sd if sd > 0 else 1.0
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit((x - self.mu) / self.sd, y)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return self.lr.predict_proba((x - self.mu) / self.sd)[:, 1]

    @property
    def slope(self):
        return float(self.lr.coef_[0, 0])


def reliability_table(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append({"bin": k, "n": int(m.sum()), "mean_pred": float(p[m].mean()) if m.any() else float("nan"),
                     "frac_wrong": float(y[m].mean()) if m.any() else float("nan")})
    return rows


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-width bins (lower is better)."""
    rows = reliability_table(y, p, n_bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] / n * abs(r["frac_wrong"] - r["mean_pred"]) for r in rows if r["n"] > 0))


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def random_scores(n, seed):
    return np.random.RandomState(seed).rand(n)


def spearman(x, y):
    rho, pval = spearmanr(x, y)
    return float(rho), float(pval)
