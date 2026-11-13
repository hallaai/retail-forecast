"""Join forecasts with actuals and segments, and score method x segment x window."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics

ALL = "all"


def scored_frame(actuals: pd.DataFrame, fc: pd.DataFrame, segments: pd.Series, cutoff: int) -> pd.DataFrame:
    """Actuals (every series that exists at the cutoff) left-joined with the forecast.

    A missing forecast is an error, not something to skip silently.
    """
    df = actuals[["unique_id", "week", "y"]].merge(fc, on=["unique_id", "week"], how="left")
    missing = int(df["f"].isna().sum())
    if missing:
        raise ValueError(f"{missing} actual rows have no forecast")
    df["segment"] = df["unique_id"].map(segments)
    df["step"] = (df["week"] - cutoff).astype(int)
    return df


def score(df: pd.DataFrame, scale: pd.Series) -> list[dict]:
    """Metrics for every segment plus the total. df: unique_id, y, f, q10..q90, segment."""
    rows = []
    for seg, part in [(ALL, df), *df.groupby("segment")]:
        y, f = part["y"].to_numpy(), part["f"].to_numpy()
        has_q = part["q10"].notna().all()
        quantile_fc = {0.1: part["q10"].to_numpy(), 0.5: part["q50"].to_numpy(), 0.9: part["q90"].to_numpy()}
        rows.append(
            {
                "segment": seg,
                "n_series": int(part["unique_id"].nunique()),
                "volume": float(y.sum()),
                "wape": metrics.wape(y, f),
                "bias": metrics.bias(y, f),
                "mase": metrics.mase(part, "f", scale),
                "swql": metrics.scaled_pinball(y, quantile_fc) if has_q else np.nan,
                "coverage80": metrics.coverage(y, part["q10"].to_numpy(), part["q90"].to_numpy()) if has_q else np.nan,
            }
        )
    return rows
