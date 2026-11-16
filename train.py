"""End-to-end credit risk pipeline.

    python train.py

Produces: reports/metrics.json, reports/*.csv and reports/figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, XGBClassifier

from src import data, evaluate, scorecard, woe

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
SEED = 42

# ---- chart style (matches the portfolio site) ----
BG, PANEL, GRID, TEXT, MUTED = "#0b1220", "#131f36", "#22314f", "#e6ebf5", "#94a3c0"
GREEN, AMBER, RED = "#34d399", "#fbbf24", "#f87171"
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "axes.grid": True,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "axes.axisbelow": True,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=160)
    plt.close(fig)


def pretty(name: str) -> str:
    return name.replace("_", " ").replace("6m", "(6m)").replace("5m", "(5m)").replace("3m", "(3m)").capitalize()


def main():
    df = data.load_dataset()
    y = df.pop(data.TARGET)
    X = df
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    print(f"Rows: {len(X):,} | default rate: {y.mean():.1%} | train {len(X_tr):,} / test {len(X_te):,}")

    # ================= 1. WoE scorecard =================
    binnings = woe.fit_all(X_tr, y_tr)
    ivs = pd.Series({f: b.iv for f, b in binnings.items()}).sort_values(ascending=False)
    W_tr, W_te = woe.woe_transform(X_tr, binnings), woe.woe_transform(X_te, binnings)

    selected = scorecard.select_features(W_tr, ivs)
    logit, feats = scorecard.fit_logit(W_tr, y_tr, selected)
    scale = scorecard.ScoreScale()
    points = scorecard.build_points_table(logit, feats, binnings, scale)
    pd_sc = logit.predict_proba(W_te[feats])[:, 1]
    scores_te = scorecard.score(W_te, logit, feats, scale)
    m_sc = evaluate.metrics(y_te, pd_sc)
    print("Scorecard features:", feats)
    print("Scorecard:", m_sc)

    # ================= 2. XGBoost challenger =================
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)
    xgb = XGBClassifier(n_estimators=2000, learning_rate=0.02, max_depth=4, min_child_weight=20,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=5, eval_metric="auc",
                        early_stopping_rounds=100, random_state=SEED, n_jobs=-1)
    xgb.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    pd_xgb = xgb.predict_proba(X_te)[:, 1]
    m_xgb = evaluate.metrics(y_te, pd_xgb)
    print("XGBoost:", m_xgb, "| trees:", xgb.best_iteration + 1)

    # ================= 3. Business view =================
    bands = evaluate.score_bands(scores_te, y_te)
    trade = evaluate.approval_tradeoff(scores_te, y_te)
    base_bad = float(y_te.mean())
    at80 = trade.iloc[(trade["approval_rate"] - 0.80).abs().argmin()]
    reduction = 1 - at80.bad_rate_approved / base_bad

    # ================= 4. Charts =================
    # IV ranking
    fig, ax = plt.subplots(figsize=(8, 5.2))
    iv_sorted = ivs.sort_values()
    colors = [GREEN if f in feats else MUTED for f in iv_sorted.index]
    ax.barh([pretty(f) for f in iv_sorted.index], iv_sorted.values, color=colors)
    ax.axvline(0.02, color=AMBER, ls="--", lw=1)
    ax.set_title("Information Value by feature")
    ax.set_xlabel("Information Value (green = used in scorecard)")
    ax.grid(axis="y", visible=False)
    save(fig, "iv_ranking.png")

    # ROC
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for p, m, lbl, c in [(pd_sc, m_sc, "WoE scorecard", GREEN), (pd_xgb, m_xgb, "XGBoost", AMBER)]:
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, color=c, lw=2.2, label=f"{lbl}  AUC {m['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_title("ROC curve (test set)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    save(fig, "roc_curve.png")

    # Score distribution
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(np.percentile(scores_te, 0.5), np.percentile(scores_te, 99.5), 45)
    ax.hist(scores_te[y_te.values == 0], bins=bins, density=True, alpha=0.75, color=GREEN, label="Repaid")
    ax.hist(scores_te[y_te.values == 1], bins=bins, density=True, alpha=0.75, color=RED, label="Defaulted")
    ax.set_title(f"Score distribution: good vs bad borrowers  (KS = {m_sc['ks']:.2f})")
    ax.set_xlabel("Credit score")
    ax.set_yticks([])
    ax.legend()
    save(fig, "score_distribution.png")

    # Default rate by score band
    fig, ax = plt.subplots(figsize=(8, 4.6))
    lbls = [f"{int(r.min_score)}–{int(r.max_score)}" for r in bands.itertuples()]
    bars = ax.bar(lbls, bands["default_rate"] * 100, color=GREEN)
    for b, v in zip(bars, bands["default_rate"]):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=9, color=TEXT)
    ax.axhline(base_bad * 100, color=AMBER, ls="--", lw=1)
    ax.text(len(lbls) - 0.5, base_bad * 100 + 1.5, f"portfolio avg {base_bad:.0%}", color=AMBER, ha="right", fontsize=9)
    ax.set_title("Default rate by score decile")
    ax.set_xlabel("Score band (low score = high risk)")
    ax.set_ylabel("Default rate (%)")
    ax.tick_params(axis="x", rotation=35, labelsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, "score_bands.png")

    # Approval trade-off
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(trade["approval_rate"] * 100, trade["bad_rate_approved"] * 100, color=GREEN, lw=2.4)
    ax.axhline(base_bad * 100, color=MUTED, ls="--", lw=1)
    ax.scatter([80], [at80.bad_rate_approved * 100], color=AMBER, zorder=5, s=50)
    ax.annotate(f"Approve 80% → default rate {at80.bad_rate_approved:.1%}\n(cut-off score {int(at80.cutoff)})",
                (80, at80.bad_rate_approved * 100), xytext=(-230, 30), textcoords="offset points",
                color=AMBER, fontsize=10, arrowprops=dict(arrowstyle="->", color=AMBER))
    ax.set_title("Risk appetite: approval rate vs. default rate of approved book")
    ax.set_xlabel("Approval rate (%)")
    ax.set_ylabel("Default rate among approved (%)")
    save(fig, "approval_tradeoff.png")

    # Calibration
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for p, lbl, c in [(pd_sc, "WoE scorecard", GREEN), (pd_xgb, "XGBoost", AMBER)]:
        frac, mean = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", color=c, lw=2, label=lbl)
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    lim = max(pd_sc.max(), pd_xgb.max()) * 1.02
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title("Calibration: predicted vs. actual PD")
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Observed default rate")
    ax.legend(loc="upper left")
    save(fig, "calibration.png")

    # SHAP
    sample = X_te.sample(2000, random_state=SEED)
    # XGBoost computes exact TreeSHAP values natively (avoids shap/xgboost 3 version clash)
    contribs = xgb.get_booster().predict(DMatrix(sample), pred_contribs=True)
    sv = shap.Explanation(values=contribs[:, :-1], base_values=contribs[:, -1],
                          data=sample.to_numpy(), feature_names=[pretty(f) for f in sample.columns])
    mean_abs_shap = pd.Series(np.abs(contribs[:, :-1]).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    plt.figure(figsize=(8.5, 5.6))
    shap.plots.beeswarm(sv, max_display=12, show=False, color_bar=True)
    fig = plt.gcf()
    for a in fig.axes:
        a.set_facecolor(PANEL)
        a.tick_params(colors=MUTED)
        a.xaxis.label.set_color(TEXT)
        a.yaxis.label.set_color(TEXT)
        for t in a.get_yticklabels():
            t.set_color(TEXT)
    fig.axes[0].set_title("XGBoost drivers of default risk (SHAP)", loc="left", color=TEXT, fontweight="bold")
    fig.set_size_inches(8.5, 5.6)
    save(fig, "shap_summary.png")

    # ================= 5. Save outputs =================
    points.to_csv(REPORTS / "scorecard_points.csv", index=False)
    pd.concat([b.table.assign(feature=f) for f, b in binnings.items()]).to_csv(REPORTS / "woe_bins.csv", index=False)
    bands.to_csv(REPORTS / "score_bands.csv", index=False)
    results = {
        "dataset": {"rows": int(len(X)), "default_rate": round(float(y.mean()), 4),
                    "train_rows": int(len(X_tr)), "test_rows": int(len(X_te))},
        "scorecard": {**m_sc, "features": feats,
                      "scale": {"base_score": scale.base_score, "base_odds": scale.base_odds, "pdo": scale.pdo}},
        "xgboost": {**m_xgb, "trees": int(xgb.best_iteration + 1),
                    "top_shap_drivers": {k: round(float(v), 4) for k, v in mean_abs_shap.head(5).items()}},
        "information_value": {k: round(float(v), 4) for k, v in ivs.items()},
        "business": {"portfolio_default_rate": round(base_bad, 4),
                     "approval_rate": 0.80, "cutoff_score": int(at80.cutoff),
                     "default_rate_approved": round(float(at80.bad_rate_approved), 4),
                     "relative_reduction": round(float(reduction), 4),
                     "top_decile_default_rate": round(float(bands["default_rate"].iloc[-1]), 4),
                     "bottom_decile_default_rate": round(float(bands["default_rate"].iloc[0]), 4)},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["business"], indent=2))


if __name__ == "__main__":
    main()
