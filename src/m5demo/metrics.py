"""Forecast accuracy metrics. Pure functions over numpy arrays / DataFrames.

* WAPE   = sum|y - f| / sum|y|                        (volume-weighted; 0 = perfect)
* Bias   = sum(f - y) / sum(y)                        (+ = over-forecast)
* MASE   = mean over series of  MAE_test / MAE_naive_in_sample   (each series counts equally)
* Pinball(q) = mean of max(q (y - f_q), (q - 1)(y - f_q)); reported scaled by mean|y|
  so it reads like WAPE ("sWQL"). Pinball at q=0.5 equals half the absolute error.
* Coverage = share of actuals inside [q0.1, q0.9]      (target 80%)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y: np.ndarray, f: np.ndarray) -> float:
    denom = np.abs(y).sum()
    return float(np.abs(y - f).sum() / denom) if denom > 0 else float("nan")


def bias(y: np.ndarray, f: np.ndarray) -> float:
    denom = np.abs(y).sum()
    return float((f - y).sum() / denom) if denom > 0 else float("nan")


def pinball(y: np.ndarray, fq: np.ndarray, q: float) -> np.ndarray:
    diff = y - fq
    return np.maximum(q * diff, (q - 1) * diff)


def scaled_pinball(y: np.ndarray, quantile_fc: dict[float, np.ndarray]) -> float:
    """Mean over quantiles of total pinball / total |y| (scale-free, volume-weighted)."""
    denom = np.abs(y).sum()
    if denom <= 0 or not quantile_fc:
        return float("nan")
    losses = [pinball(y, fq, q).sum() / denom for q, fq in quantile_fc.items()]
    return float(2 * np.mean(losses))  # x2 so that the q=0.5 term alone equals WAPE


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(((y >= lo) & (y <= hi)).mean()) if len(y) else float("nan")


def naive_scale(train: pd.DataFrame) -> pd.Series:
    """In-sample MAE of the lag-1 naive forecast per series (MASE denominator).

    Lag-1 (not lag-52) so that short / new series still get a scale.
    Series with a zero scale (constant history) get NaN and are excluded from MASE.
    """
    diffs = train.sort_values(["unique_id", "week"]).groupby("unique_id")["y"].diff().abs()
    scale = diffs.groupby(train["unique_id"]).mean()
    return scale.where(scale > 0)


def mase(frame: pd.DataFrame, fc_col: str, scale: pd.Series) -> float:
    """frame: unique_id, y, <fc_col>. Mean of per-series scaled MAE over series with a valid scale."""
    mae = (frame["y"] - frame[fc_col]).abs().groupby(frame["unique_id"]).mean()
    ratio = (mae / scale.reindex(mae.index)).dropna()
    return float(ratio.mean()) if len(ratio) else float("nan")
