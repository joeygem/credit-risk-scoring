"""Load the UCI 'Default of Credit Card Clients' dataset and engineer behavioural features."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "default of credit card clients.xls"

TARGET = "default"

# Protected / demographic attributes are deliberately excluded from modelling
# (fair-lending practice: decisions should rest on credit behaviour, not who the borrower is).
EXCLUDED = ["SEX", "MARRIAGE", "EDUCATION", "AGE"]

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]


def load_raw(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={"default payment next month": TARGET}).drop(columns="ID")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Build credit-bureau-style behavioural features from 6 months of history."""
    out = pd.DataFrame(index=df.index)
    limit = df["LIMIT_BAL"].astype(float)

    # Delinquency: PAY_x >= 1 means months past due; -2/-1/0 mean no delay.
    delinq = df[PAY_STATUS].clip(lower=0)
    out["limit_bal"] = limit
    out["months_late_latest"] = delinq["PAY_0"]
    out["max_months_late_6m"] = delinq.max(axis=1)
    out["n_months_delinquent_6m"] = (delinq > 0).sum(axis=1)
    out["n_months_delinquent_3m"] = (delinq[PAY_STATUS[:3]] > 0).sum(axis=1)

    # Utilisation: balance relative to credit limit.
    util = df[BILLS].div(limit, axis=0)
    out["utilisation_latest"] = util["BILL_AMT1"]
    out["utilisation_avg_6m"] = util.mean(axis=1)
    out["utilisation_max_6m"] = util.max(axis=1)
    out["balance_trend_6m"] = (df["BILL_AMT1"] - df["BILL_AMT6"]) / limit

    # Repayment behaviour: PAY_AMTi pays the statement BILL_AMT(i+1).
    paid = df[PAYMENTS[:5]].to_numpy(dtype=float)
    billed = df[BILLS[1:]].to_numpy(dtype=float)
    ratio = np.where(billed > 0, paid / np.maximum(billed, 1), 1.0)
    ratio = np.clip(ratio, 0, 2)
    out["payment_ratio_latest"] = ratio[:, 0]
    out["payment_ratio_avg_5m"] = ratio.mean(axis=1)
    out["n_zero_payments_6m"] = (df[PAYMENTS] == 0).sum(axis=1)
    out["avg_payment_to_limit"] = df[PAYMENTS].mean(axis=1) / limit

    out[TARGET] = df[TARGET].astype(int)
    return out


def load_dataset() -> pd.DataFrame:
    return engineer(load_raw())
