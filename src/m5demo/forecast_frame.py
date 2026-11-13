"""The one forecast schema every method writes.

unique_id, week, f (point), q10, q50, q90 (NaN when the method has no distribution).
All values are clipped at 0: demand cannot be negative. A non-finite quantile (StatsForecast can
return inf when a series' residual variance divides by zero) is replaced by the point forecast.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
COLUMNS = ["unique_id", "week", "f", "q10", "q50", "q90"]
Q_COLS = ["q10", "q50", "q90"]


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +-inf quantiles by the point forecast (only where the method has quantiles at all)."""
    out = df.copy()
    for col in Q_COLS:
        bad = np.isinf(out[col].to_numpy(dtype="float64"))
        if bad.any():
            log.warning("%d non-finite %s values replaced by the point forecast", int(bad.sum()), col)
            out.loc[bad, col] = out.loc[bad, "f"]
    return out


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in Q_COLS:
        if col not in out:
            out[col] = np.nan
    out = sanitize(out)
    num = ["f", *Q_COLS]
    out[num] = out[num].astype("float32").clip(lower=0)
    out["week"] = out["week"].astype("int16")
    return out[COLUMNS].sort_values(["unique_id", "week"]).reset_index(drop=True)


def ds_to_week(ds: pd.Series, calendar: pd.DataFrame) -> pd.Series:
    mapping = dict(zip(calendar["ds"], calendar["week"]))
    return ds.map(mapping)
