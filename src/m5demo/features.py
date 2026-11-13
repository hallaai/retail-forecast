"""Exogenous features and the ratio-target transform. Pure functions over DataFrames.

Prices in M5 are published ahead of time (planned shelf prices), so price features
for the forecast weeks are treated as known in advance, like a planned markdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATIC_COLS = ["dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLS = [
    "event_sporting",
    "event_cultural",
    "event_national",
    "event_religious",
    "is_thanksgiving",
    "is_christmas",
    "is_superbowl",
    "is_easter",
]
EXOG_COLS = ["price", "price_rel_max52", "price_change", "price_rel_dept", "snap_days", *CALENDAR_COLS]


def build_exog(panel: pd.DataFrame, static: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Per (unique_id, week): price features + calendar events + SNAP days of the store's state."""
    df = panel[["unique_id", "week", "price"]].merge(static[["unique_id", *STATIC_COLS]], on="unique_id")
    df = df.sort_values(["unique_id", "week"])
    by_id = df.groupby("unique_id")["price"]
    df["price_rel_max52"] = df["price"] / by_id.transform(lambda s: s.rolling(52, min_periods=1).max())
    df["price_change"] = (df["price"] / by_id.shift(1) - 1).fillna(0.0)
    dept_week = df.groupby(["dept_id", "store_id", "week"])["price"].transform("mean")
    df["price_rel_dept"] = df["price"] / dept_week

    cal = calendar.set_index("week")
    snap = cal[["snap_CA", "snap_TX", "snap_WI"]].rename(columns=lambda c: c.split("_")[1])
    snap_long = snap.stack().rename("snap_days").reset_index().rename(columns={"level_1": "state_id"})
    df = df.merge(snap_long, on=["week", "state_id"], how="left")
    df = df.merge(cal[CALENDAR_COLS].reset_index(), on="week", how="left")
    float_cols = ["price", "price_rel_max52", "price_change", "price_rel_dept"]
    df[float_cols] = df[float_cols].astype("float32")
    for col in STATIC_COLS:
        df[col] = df[col].astype("category")
    return df


PRICE_COLS = ["price", "price_rel_max52", "price_change", "price_rel_dept"]


def freeze_prices(exog: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Ablation: future weeks see the last price known at the cutoff (no planned price changes)."""
    last = exog[exog["week"] <= cutoff].sort_values("week").groupby("unique_id")[PRICE_COLS].last()
    future = exog["week"] > cutoff
    out = exog.copy()
    frozen = last.reindex(out.loc[future, "unique_id"]).to_numpy()
    out.loc[future, PRICE_COLS] = frozen
    out.loc[future, "price_change"] = 0.0
    return out


def trailing_mean_scale(train: pd.DataFrame, cutoff: int, window: int, eps: float) -> pd.Series:
    """Mean of y over the `window` weeks up to the cutoff (fewer for young series), plus eps."""
    recent = train[train["week"] > cutoff - window]
    return recent.groupby("unique_id")["y"].mean().astype("float64") + eps


def to_ratio(train: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    return train.assign(y=train["y"] / train["unique_id"].map(scale).to_numpy())


def from_ratio(fc: pd.DataFrame, scale: pd.Series, cols: list[str]) -> pd.DataFrame:
    factor = fc["unique_id"].map(scale).to_numpy()
    return fc.assign(**{c: fc[c].to_numpy() * factor for c in cols})


def week_of_year(dates: pd.DatetimeIndex) -> np.ndarray:
    return dates.isocalendar().week.to_numpy().astype(np.int16)
