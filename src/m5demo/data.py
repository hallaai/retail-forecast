"""Load raw M5 CSVs and build the weekly panel (I/O lives here, nowhere else).

Output tables
-------------
panel.parquet   unique_id, week, ds, y, price  (dense from each series' first sale to the end)
static.parquet  unique_id, item_id, dept_id, cat_id, store_id, state_id
calendar.parquet week, ds, event_* counts, snap_CA/TX/WI (days in week)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
EVENT_TYPES = ("Sporting", "Cultural", "National", "Religious")


def load_calendar(raw_dir: Path) -> pd.DataFrame:
    cal = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    cal["d_idx"] = np.arange(len(cal))
    return cal


def complete_weeks(cal: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Weeks fully covered by the sales history; the partial tail week is dropped."""
    days = cal.iloc[:n_days]
    counts = days.groupby("wm_yr_wk").size()
    full = counts[counts == 7].index
    weeks = (
        days[days["wm_yr_wk"].isin(full)]
        .groupby("wm_yr_wk", as_index=False)
        .agg(ds=("date", "min"), first_day=("d_idx", "min"))
        .sort_values("wm_yr_wk")
        .reset_index(drop=True)
    )
    weeks["week"] = np.arange(len(weeks))
    return weeks


def weekly_sales(sales: pd.DataFrame, n_weeks: int) -> np.ndarray:
    """Sum consecutive blocks of 7 day-columns. Day 1 is a Saturday = week start."""
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    daily = sales[day_cols[: n_weeks * 7]].to_numpy(dtype=np.float32)
    return daily.reshape(len(sales), n_weeks, 7).sum(axis=2)


def weekly_calendar(cal: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    days = cal.merge(weeks[["wm_yr_wk", "week", "ds"]], on="wm_yr_wk")
    out = days.groupby(["week", "ds"], as_index=False)[["snap_CA", "snap_TX", "snap_WI"]].sum()
    for etype in EVENT_TYPES:
        hit = (days["event_type_1"] == etype) | (days["event_type_2"] == etype)
        out[f"event_{etype.lower()}"] = hit.groupby(days["week"]).sum().reindex(out["week"]).to_numpy()
    for name in ("Thanksgiving", "Christmas", "SuperBowl", "Easter"):
        hit = (days["event_name_1"] == name) | (days["event_name_2"] == name)
        out[f"is_{name.lower()}"] = hit.groupby(days["week"]).max().reindex(out["week"]).astype(int).to_numpy()
    return out


def build_panel(raw_dir: Path = config.RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cal = load_calendar(raw_dir)
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    n_days = sum(c.startswith("d_") for c in sales.columns)
    weeks = complete_weeks(cal, n_days)
    n_weeks = len(weeks)

    static = sales[ID_COLS].copy()
    static.insert(0, "unique_id", static["item_id"] + "_" + static["store_id"])
    y = weekly_sales(sales, n_weeks)

    first_sale = np.where((y > 0).any(axis=1), (y > 0).argmax(axis=1), n_weeks)
    long = _dense_long(static["unique_id"].to_numpy(), y, first_sale)
    long = long.merge(weeks[["week", "ds", "wm_yr_wk"]], on="week")

    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices["unique_id"] = prices["item_id"] + "_" + prices["store_id"]
    long = long.merge(prices[["unique_id", "wm_yr_wk", "sell_price"]], on=["unique_id", "wm_yr_wk"], how="left")
    long = long.rename(columns={"sell_price": "price"}).drop(columns="wm_yr_wk")
    long = long.sort_values(["unique_id", "week"]).reset_index(drop=True)
    long["price"] = long.groupby("unique_id")["price"].transform(lambda s: s.ffill().bfill())

    return long, static, weekly_calendar(cal, weeks)


def _dense_long(ids: np.ndarray, y: np.ndarray, first_sale: np.ndarray) -> pd.DataFrame:
    """Dense grid from each series' first sale week to the last complete week (zeros kept)."""
    _n_series, n_weeks = y.shape
    keep = np.arange(n_weeks)[None, :] >= first_sale[:, None]
    rows, cols = np.nonzero(keep)
    return pd.DataFrame({"unique_id": ids[rows], "week": cols.astype(np.int16), "y": y[rows, cols]})


def save_processed(out_dir: Path = config.PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel, static, calendar = build_panel()
    panel.to_parquet(out_dir / "panel.parquet", index=False)
    static.to_parquet(out_dir / "static.parquet", index=False)
    calendar.to_parquet(out_dir / "calendar.parquet", index=False)


def load_processed(out_dir: Path = config.PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(out_dir / "panel.parquet"),
        pd.read_parquet(out_dir / "static.parquet"),
        pd.read_parquet(out_dir / "calendar.parquet"),
    )
