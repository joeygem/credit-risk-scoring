"""Credit-risk evaluation metrics."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score


def metrics(y, pd_hat) -> dict:
    y = np.asarray(y)
    auc = roc_auc_score(y, pd_hat)
    ks = ks_2samp(pd_hat[y == 1], pd_hat[y == 0]).statistic
    return {"auc": round(auc, 4), "gini": round(2 * auc - 1, 4),
            "ks": round(float(ks), 4), "brier": round(brier_score_loss(y, pd_hat), 4)}


def score_bands(scores, y, n_bands=10) -> pd.DataFrame:
    df = pd.DataFrame({"score": scores, "y": np.asarray(y)})
    df["band"] = pd.qcut(df["score"], n_bands, duplicates="drop")
    t = df.groupby("band", observed=True).agg(count=("y", "size"), defaults=("y", "sum"),
                                              min_score=("score", "min"), max_score=("score", "max"))
    t["default_rate"] = t["defaults"] / t["count"]
    return t.reset_index(drop=True)


def approval_tradeoff(scores, y) -> pd.DataFrame:
    """For each approval rate, the default rate among approved (approve the highest scores)."""
    order = np.argsort(-np.asarray(scores))
    y_sorted = np.asarray(y)[order]
    n = len(y_sorted)
    approved = np.arange(1, n + 1)
    bad_rate = np.cumsum(y_sorted) / approved
    rates = np.linspace(0.05, 1.0, 96)
    idx = np.clip((rates * n).astype(int) - 1, 0, n - 1)
    cutoffs = np.asarray(scores)[order][idx]
    return pd.DataFrame({"approval_rate": rates, "bad_rate_approved": bad_rate[idx], "cutoff": cutoffs})
