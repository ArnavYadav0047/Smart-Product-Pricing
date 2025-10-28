# utils_metrics.py
# Purpose: Centralize SMAPE/diagnostics and enforce submission schema.

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error

def smape(y_true, y_pred, eps=1e-9):
    """Compute SMAPE (%) with small epsilon for numerical stability."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

def smape_surrogate(y_true, y_pred, alpha=1e-3):
    """Smooth SMAPE-like objective for diagnostics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred) + alpha) / 2.0
    return np.mean(np.abs(y_pred - y_true) / denom)

def decile_smape(y_true, y_pred, n_bins=10):
    """SMAPE within price deciles to ensure balanced accuracy."""
    df = pd.DataFrame({"y": y_true, "yp": y_pred})
    df["bin"] = pd.qcut(df["y"].rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    out = []
    for b in sorted(df["bin"].dropna().unique()):
        sub = df[df["bin"] == b]
        out.append(float(smape(sub["y"].values, sub["yp"].values)))
    return out

def calibration_bins(y_true, y_pred, n_bins=10):
    """Mean predicted vs mean actual per predicted-price bin."""
    df = pd.DataFrame({"y": y_true, "yp": y_pred})
    df["bin"] = pd.qcut(df["yp"].rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    bins = []
    for b in sorted(df["bin"].dropna().unique()):
        sub = df[df["bin"] == b]
        bins.append({
            "bin": int(b),
            "mean_pred": float(sub["yp"].mean()),
            "mean_true": float(sub["y"].mean()),
            "count": int(len(sub))
        })
    return bins

def quantile_error_bands(y_true, y_pred):
    """Q10/Q50/Q90 of APE under SMAPE denominator."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred) + 1e-9) / 2.0)
    return {
        "q10": float(np.quantile(ape, 0.10) * 100.0),
        "q50": float(np.quantile(ape, 0.50) * 100.0),
        "q90": float(np.quantile(ape, 0.90) * 100.0),
    }

def residual_stats(y_true, y_pred):
    """Residual mean/std/skew as bias/variance checks."""
    res = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return {
        "mean": float(np.mean(res)),
        "std": float(np.std(res)),
        "skew": float((np.mean((res - res.mean())**3) / (np.std(res) + 1e-9)**3))
    }

def secondary_metrics(y_true, y_pred):
    """MAE/MedAE on price; log-MAE and R² on log-price as diagnostics."""
    out = {}
    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    out["medae"] = float(median_absolute_error(y_true, y_pred))
    y = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    mask = (y > 0) & (yp > 0)
    if mask.any():
        out["log_mae"] = float(np.mean(np.abs(np.log(y[mask]) - np.log(yp[mask]))))
        out["r2_logy"] = float(r2_score(np.log(y[mask]), np.log(yp[mask])))
    else:
        out["log_mae"] = None
        out["r2_logy"] = None
    return out

def validate_submission_schema(test_df, sub_df, sample_sub_df):
    """Prevent evaluation rejection: exact headers, order, row count, positive numeric prices."""
    assert list(sub_df.columns) == ["sample_id", "price"], "Submission columns must be exactly ['sample_id','price']"
    assert len(sub_df) == len(test_df), "Submission rows must equal test rows"
    assert sub_df["sample_id"].tolist() == test_df["sample_id"].tolist(), "Submission order must match test order"
    prices = sub_df["price"].astype(str)
    parsed = pd.to_numeric(prices, errors="coerce")
    assert parsed.notna().all(), "All prices must be numeric"
    assert (parsed > 0).all(), "All prices must be > 0"
    assert list(sample_sub_df.columns) == ["sample_id","price"], "Sample output header mismatch"
