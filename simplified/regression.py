"""Fit the IPMVP Option C daily regression on IPMVP_C_data.csv.

Model (ordinary least squares):
    E(i) = b0 + sum_{j=2..7} b_j W_j(i) + b_CDD CDD(i) + b_HDD HDD(i) + b_C Controlled(i)

W_j(i) is 1 when day i is weekday j (Sunday=1 ... Saturday=7). Sunday (j=1) is the
reference day and has no dummy, because all seven dummies together would duplicate
the intercept and the fit would have no unique solution. Each b_j is therefore
"kWh difference from a Sunday".

b_C is the average daily change in consumption during the controlled period,
after adjusting for weekday and weather (negative = savings).

Usage:
    py regression.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE / "IPMVP_C_data.csv"
COEF_CSV = HERE / "regression_coefficients.csv"
FITTED_CSV = HERE / "regression_fitted.csv"

DAY_NAMES = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
             5: "Thursday", 6: "Friday", 7: "Saturday"}


def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={
        "Summed Energy Used": "E",
        "Heating Degree Days": "HDD",
        "Cooling Degree Days": "CDD",
    })
    before = len(df)
    df = df.dropna(subset=["E", "HDD", "CDD", "Controlled"]).reset_index(drop=True)
    dropped = before - len(df)
    # Sunday=1 ... Saturday=7 (pandas dayofweek is Monday=0 ... Sunday=6).
    df["weekday"] = (df["date"].dt.dayofweek + 1) % 7 + 1
    return df, dropped


def design_matrix(df, include_controlled=True):
    cols = {"b0 (intercept)": np.ones(len(df))}
    for j in range(2, 8):
        cols[f"b{j} W{j} {DAY_NAMES[j]}"] = (df["weekday"] == j).astype(float).to_numpy()
    cols["b_CDD"] = df["CDD"].to_numpy(dtype=float)
    cols["b_HDD"] = df["HDD"].to_numpy(dtype=float)
    if include_controlled:
        cols["b_C Controlled"] = df["Controlled"].to_numpy(dtype=float)
    names = list(cols)
    return np.column_stack([cols[n] for n in names]), names


def fit_ols(X, y):
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    dof = n - k
    sse = float(resid @ resid)
    sigma2 = sse / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    # Two-sided p-values with a normal approximation (dof is large here).
    p = np.array([math.erfc(abs(v) / math.sqrt(2)) for v in t])
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst
    stats = {
        "n": n,
        "parameters": k,
        "R2": r2,
        "adj_R2": 1 - (1 - r2) * (n - 1) / dof,
        "RMSE": math.sqrt(sigma2),
        "CV(RMSE) %": 100 * math.sqrt(sigma2) / y.mean(),
    }
    return beta, se, t, p, fitted, stats


def main():
    df, dropped = load_data(DATA_CSV)
    if df["Controlled"].nunique() < 2:
        raise SystemExit("Controlled column has only one value; cannot estimate b_C.")

    X, names = design_matrix(df)
    y = df["E"].to_numpy(dtype=float)
    beta, se, t, p, fitted, stats = fit_ols(X, y)

    coef = pd.DataFrame({"term": names, "coefficient": beta, "std_error": se,
                         "t_stat": t, "p_value": p})
    coef.to_csv(COEF_CSV, index=False, float_format="%.6g")

    out = df[["date", "E", "HDD", "CDD", "Controlled", "weekday"]].copy()
    out["fitted"] = fitted
    out["residual"] = y - fitted
    out.to_csv(FITTED_CSV, index=False, float_format="%.2f", date_format="%Y-%m-%d")

    print(f"Rows used: {stats['n']}" + (f" ({dropped} dropped for blank values)" if dropped else ""))
    print(f"Baseline days: {(df['Controlled'] == 0).sum()}, controlled days: {(df['Controlled'] == 1).sum()}\n")
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 120):
        print(coef.to_string(index=False))
    print()
    for key in ("R2", "adj_R2", "RMSE", "CV(RMSE) %"):
        print(f"{key:>11}: {stats[key]:,.4f}")

    b_c, se_c = beta[-1], se[-1]
    n_ctrl = int((df["Controlled"] == 1).sum())
    print(f"\nControlled-period effect: {b_c:,.1f} kWh/day "
          f"(95% CI {b_c - 1.96 * se_c:,.1f} to {b_c + 1.96 * se_c:,.1f})")
    print(f"Over {n_ctrl} controlled days: {b_c * n_ctrl:,.0f} kWh "
          f"({'savings' if b_c < 0 else 'increase'})")
    print(f"\nSaved {COEF_CSV.name} and {FITTED_CSV.name}")


if __name__ == "__main__":
    main()
