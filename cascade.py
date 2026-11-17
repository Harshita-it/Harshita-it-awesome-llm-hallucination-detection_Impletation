"""RQ3: token-gated cascade, evaluated OFFLINE from stored scores (protocol Section 10)."""
import numpy as np


def threshold(s_cal, q):
    """q-quantile of the calibration gate scores. q<=0 escalates everything, q>=1 escalates nothing."""
    if q <= 0:
        return -np.inf
    if q >= 1:
        return np.inf
    return float(np.quantile(np.asarray(s_cal, dtype=float), q))


def cascade_risk(q, s_cal, s_test, se_test, platt_tok, platt_se):
    """Primary cascade: risk = Platt(SE) for escalated questions, Platt(token score) otherwise.
    Returns (risk, escalated_mask)."""
    s_test = np.asarray(s_test, dtype=float)
    esc = s_test >= threshold(s_cal, q)
    risk = np.where(esc, platt_se.predict(se_test), platt_tok.predict(s_test))
    return risk, esc


def rank_rule_risk(esc, s_test, se_test):
    """Secondary cascade: all escalated questions rank above all accepted ones; escalated ordered by SE (ties by token
    score), accepted by token score. Returned values are comparable ranks in (0, 2)."""
    esc = np.asarray(esc, dtype=bool)
    s = np.asarray(s_test, dtype=float)
    se = np.asarray(se_test, dtype=float)
    out = np.zeros(len(s))

    def ranks(primary, secondary):
        order = np.lexsort((secondary, primary))
        r = np.empty(len(primary))
        r[order] = np.arange(1, len(primary) + 1)
        return r / (len(primary) + 1)

    if (~esc).any():
        out[~esc] = 0.5 * ranks(s[~esc], s[~esc])
    if esc.any():
        out[esc] = 1.0 + 0.5 * ranks(se[esc], s[esc])
    return out


def cost_summary(esc, nli_calls, seconds_greedy, seconds_samples, seconds_nli, n_samples):
    """Cost of the cascade on the evaluated questions: the token stage runs for all, the SE stage only for escalated.
    Counts are exact; seconds are the measured per-stage times summed accordingly."""
    esc = np.asarray(esc, dtype=bool)
    n = len(esc)
    nli = np.asarray(nli_calls, dtype=float)
    sg = np.asarray(seconds_greedy, dtype=float)
    ss = np.asarray(seconds_samples, dtype=float)
    sn = np.asarray(seconds_nli, dtype=float)
    return {
        "escalated_fraction": float(esc.mean()),
        "extra_generations_per_question": float(n_samples * esc.sum() / n),
        "nli_calls_per_question": float(nli[esc].sum() / n),
        "seconds_per_question": float((sg.sum() + (ss + sn)[esc].sum()) / n),
    }
