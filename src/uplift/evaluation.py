"""Evaluate CATE models by ranking, not per-user error."""

import numpy as np
import pandas as pd

from uplift.ate import compute_ate


def qini_curve(df: pd.DataFrame, predicted_cate, outcome: str,
               treatment_col: str = "treatment", n_points: int = 100) -> pd.DataFrame:
    """Cumulative incremental outcomes when targeting the top-k users by predicted CATE.

    At each k: qini = (mean outcome treated - mean outcome control among the top k) * k,
    i.e. the measured ATE of the targeted group times its size. `random` is the straight
    line from 0 to the overall ATE * N, what you'd get by targeting in random order.
    """
    t = df[treatment_col].to_numpy()
    y = df[outcome].to_numpy()
    order = np.argsort(-np.asarray(predicted_cate), kind="stable")
    t, y = t[order], y[order]

    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))

    n = len(df)
    ks = np.unique(np.linspace(1, n, n_points).astype(int))
    idx = ks - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        ate_k = cum_yt[idx] / cum_t[idx] - cum_yc[idx] / cum_c[idx]
    qini = np.nan_to_num(ate_k * ks)

    overall = qini[-1]
    frac = ks / n
    return pd.DataFrame({"fraction": np.r_[0.0, frac], "qini": np.r_[0.0, qini],
                         "random": np.r_[0.0, frac * overall]})


def auuc(curve: pd.DataFrame) -> float:
    """Area between the Qini curve and the random-targeting line (Qini coefficient).
    Positive = ranking beats random. Units: incremental outcomes x fraction of population."""
    gap = curve["qini"] - curve["random"]
    return float(np.trapezoid(gap, curve["fraction"]))


def decile_table(df: pd.DataFrame, predicted_cate, outcome: str,
                  treatment_col: str = "treatment", n_bins: int = 10, alpha: float = 0.05) -> pd.DataFrame:
    """Bin held-out users by predicted CATE, report the actual measured ATE within each bin.

    Decile 0 = lowest predicted CATE, decile n_bins-1 = highest. `alpha` sets each
    bin's CI level - pass alpha/n_bins (Bonferroni correction) to control the
    overall false-positive rate across all n_bins simultaneous tests, not just one.
    """
    work = df.copy()
    work["predicted_cate"] = predicted_cate
    # rank first, then cut, so tied/duplicate predicted values don't break qcut into fewer bins
    work["decile"] = pd.qcut(work["predicted_cate"].rank(method="first"), n_bins, labels=False)

    rows = []
    for d in range(n_bins):
        bucket = work[work["decile"] == d]
        result = compute_ate(bucket, outcome, treatment_col, alpha=alpha)
        rows.append({
            "decile": d,
            "n": len(bucket),
            "mean_predicted_cate": bucket["predicted_cate"].mean(),
            "actual_ate": result.ate,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
        })

    return pd.DataFrame(rows)
