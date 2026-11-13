"""Quantiles for point-only methods, calibrated on backtest residuals (split-conformal style).

For each (segment, horizon step) we take the empirical quantiles of the scaled residual
    r = (y - f) / scale,   scale = trailing 26-week mean at that window's cutoff (+ eps)
over all backtest windows, then set  q_tau = max(0, f + scale * r_tau)  in the test window.
Only backtest windows feed the calibration, so the test window stays untouched.
Pure functions; inputs are plain DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import QUANTILES

Q_COLS = {0.1: "q10", 0.5: "q50", 0.9: "q90"}


def residual_quantiles(backtest: pd.DataFrame, quantiles: tuple[float, ...] = QUANTILES) -> pd.DataFrame:
    """backtest: segment, step, y, f, scale -> table indexed by (segment, step) with one column per quantile."""
    r = (backtest["y"] - backtest["f"]) / backtest["scale"]
    grouped = r.groupby([backtest["segment"], backtest["step"]])
    table = pd.concat({Q_COLS[q]: grouped.quantile(q) for q in quantiles}, axis=1)
    return table


def apply_quantiles(fc: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """fc: unique_id, week, f, segment, step, scale -> same frame with q10/q50/q90 filled."""
    keyed = fc.merge(table, left_on=["segment", "step"], right_index=True, how="left", suffixes=("_old", ""))
    out = fc.copy()
    for col in table.columns:
        offsets = keyed[col].fillna(0.0).to_numpy()
        out[col] = np.maximum(0.0, fc["f"].to_numpy() + fc["scale"].to_numpy() * offsets).astype("float32")
    # Enforce q10 <= q50 <= q90 (independent empirical quantiles can cross after clipping).
    stacked = np.sort(out[["q10", "q50", "q90"]].to_numpy(), axis=1)
    out[["q10", "q50", "q90"]] = stacked
    return out
