"""Logistic-regression scorecard on WoE features, scaled to points."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class ScoreScale:
    base_score: float = 600   # score at base odds
    base_odds: float = 50     # good:bad odds at the base score
    pdo: float = 20           # points to double the odds

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)


def select_features(woe_X: pd.DataFrame, ivs: pd.Series, min_iv=0.02, max_corr=0.75) -> list:
    """Keep informative features, dropping the weaker of any highly correlated pair."""
    candidates = ivs[ivs >= min_iv].sort_values(ascending=False).index.tolist()
    corr = woe_X[candidates].corr().abs()
    keep = []
    for c in candidates:
        if all(corr.loc[c, k] < max_corr for k in keep):
            keep.append(c)
    return keep


def fit_logit(woe_X: pd.DataFrame, y: pd.Series, features: list) -> tuple:
    """Fit, then drop any feature whose sign contradicts WoE logic (coef must be negative)."""
    feats = list(features)
    while True:
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(woe_X[feats], y)
        coefs = pd.Series(model.coef_[0], index=feats)
        wrong = coefs[coefs > 0]
        if wrong.empty:
            return model, feats
        feats.remove(wrong.idxmax())


def build_points_table(model, feats: list, binnings: dict, scale: ScoreScale) -> pd.DataFrame:
    n = len(feats)
    b0 = model.intercept_[0]
    rows = []
    for coef, f in zip(model.coef_[0], feats):
        t = binnings[f].table
        for _, r in t.iterrows():
            # score = offset + factor * ln(odds_good) = offset - factor * logit(bad)
            pts = scale.offset / n - scale.factor * (coef * r["woe"] + b0 / n)
            rows.append({"feature": f, "bin": r["bin"], "woe": r["woe"],
                         "bad_rate": r["bad_rate"], "share": r["share"], "points": round(pts)})
    return pd.DataFrame(rows)


def score(woe_X: pd.DataFrame, model, feats: list, scale: ScoreScale) -> np.ndarray:
    logit = model.decision_function(woe_X[feats])
    return scale.offset - scale.factor * logit
