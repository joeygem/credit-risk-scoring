"""Weight-of-Evidence binning with monotonic bad-rate constraint, written from scratch.

Convention: WoE = ln(%good / %bad), so a higher WoE means a lower-risk bin.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Binning:
    feature: str
    edges: list                      # interior cut points; bins are (-inf, e1], (e1, e2], ... (ek, inf)
    table: pd.DataFrame = field(default=None)

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    def bin_index(self, x: pd.Series) -> np.ndarray:
        return np.searchsorted(np.asarray(self.edges, dtype=float), x.to_numpy(dtype=float), side="left")

    def transform(self, x: pd.Series) -> np.ndarray:
        return self.table["woe"].to_numpy()[self.bin_index(x)]

    def labels(self) -> list:
        e = [-np.inf] + list(self.edges) + [np.inf]
        fmt = lambda v: "-inf" if v == -np.inf else ("inf" if v == np.inf else f"{v:,.4g}")
        return [f"({fmt(lo)}, {fmt(hi)}]" for lo, hi in zip(e[:-1], e[1:])]


def _stats(idx: np.ndarray, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    total = np.bincount(idx, minlength=n_bins).astype(float)
    bad = np.bincount(idx, weights=y, minlength=n_bins).astype(float)
    good = total - bad
    # +0.5 smoothing avoids infinite WoE in pure bins
    pg = (good + 0.5) / (good.sum() + 0.5 * n_bins)
    pb = (bad + 0.5) / (bad.sum() + 0.5 * n_bins)
    woe = np.log(pg / pb)
    return pd.DataFrame({
        "count": total, "bad": bad, "good": good,
        "share": total / total.sum(), "bad_rate": bad / np.maximum(total, 1),
        "woe": woe, "iv": (pg - pb) * woe,
    })


def _initial_edges(x: pd.Series, max_bins: int) -> list:
    uniq = np.sort(x.dropna().unique())
    if len(uniq) <= max_bins:
        return list(uniq[:-1])
    qs = np.unique(np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1]))
    return list(qs)


def fit_binning(x: pd.Series, y: pd.Series, max_bins: int = 10, min_share: float = 0.05) -> Binning:
    y = y.to_numpy(dtype=float)
    edges = _initial_edges(x, max_bins)

    def table(edges):
        idx = np.searchsorted(np.asarray(edges, dtype=float), x.to_numpy(dtype=float), side="left")
        return _stats(idx, y, len(edges) + 1)

    # 1) merge bins that are too small into their most similar neighbour
    while edges:
        t = table(edges)
        small = t.index[t["share"] < min_share]
        if len(small) == 0:
            break
        i = small[0]
        if i == 0:
            drop = 0
        elif i == len(t) - 1:
            drop = i - 1
        else:
            left = abs(t.bad_rate[i] - t.bad_rate[i - 1])
            right = abs(t.bad_rate[i] - t.bad_rate[i + 1])
            drop = i - 1 if left <= right else i
        edges.pop(drop)

    # 2) enforce a monotonic bad-rate trend (direction from overall correlation)
    while len(edges) > 0:
        t = table(edges)
        br = t["bad_rate"].to_numpy()
        direction = np.sign(np.corrcoef(np.arange(len(br)), br)[0, 1]) if len(br) > 2 else 1
        diffs = np.diff(br) * direction
        bad_steps = np.where(diffs < 0)[0]
        if len(bad_steps) == 0:
            break
        edges.pop(bad_steps[0])

    b = Binning(feature=x.name, edges=edges)
    b.table = table(edges)
    b.table.insert(0, "bin", b.labels())
    return b


def fit_all(X: pd.DataFrame, y: pd.Series, **kw) -> dict:
    return {c: fit_binning(X[c], y, **kw) for c in X.columns}


def woe_transform(X: pd.DataFrame, binnings: dict) -> pd.DataFrame:
    return pd.DataFrame({c: binnings[c].transform(X[c]) for c in binnings}, index=X.index)


def iv_label(iv: float) -> str:
    if iv < 0.02:
        return "Not predictive"
    if iv < 0.1:
        return "Weak"
    if iv < 0.3:
        return "Medium"
    if iv < 0.5:
        return "Strong"
    return "Very strong"
