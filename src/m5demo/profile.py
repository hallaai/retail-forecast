"""What the data looks like: tables, columns, row counts and gaps. Pure functions, no I/O.

Feeds the frontend's Data tab. Everything here is computed from the processed tables, so it
describes exactly what the models were given.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

LONG_ZERO_RUN = 13  # weeks; a quarter with no sale after the first sale is treated as a gap

TABLES = {
    "panel": ("Weekly sales", "one row per series and week, from its first sale to the last complete week"),
    "static": ("Series attributes", "one row per item × store series"),
    "calendar": ("Calendar", "one row per week"),
    "exog": ("Model features", "one row per series and week; what LightGBM sees besides the sales history"),
}

COLUMNS = {
    "unique_id": "Series key: item id + store id",
    "week": "Week index, 0 = week starting 2011-01-29",
    "ds": "Saturday that starts the week",
    "y": "Units sold in the week (sum of 7 days)",
    "price": "Shelf price; weeks without a listed price take the nearest listed price",
    "item_id": "Product",
    "dept_id": "Department, e.g. FOODS_3",
    "cat_id": "Category: FOODS, HOBBIES or HOUSEHOLD",
    "store_id": "Store, e.g. CA_1",
    "state_id": "State: CA, TX or WI",
    "snap_CA": "SNAP (food-stamp) days in the week, California",
    "snap_TX": "SNAP days in the week, Texas",
    "snap_WI": "SNAP days in the week, Wisconsin",
    "snap_days": "SNAP days in the week in the series' state",
    "event_sporting": "Sporting event days in the week",
    "event_cultural": "Cultural event days in the week",
    "event_national": "National holiday days in the week",
    "event_religious": "Religious holiday days in the week",
    "is_thanksgiving": "Week contains Thanksgiving",
    "is_christmas": "Week contains Christmas (stores closed that day)",
    "is_superbowl": "Week contains the Super Bowl",
    "is_easter": "Week contains Easter",
    "price_rel_max52": "Price ÷ highest price in the last 52 weeks (below 1 = discounted)",
    "price_change": "Price change vs the previous week",
    "price_rel_dept": "Price ÷ mean price of the department in the same store and week",
}

_STATIC = {"dept_id", "cat_id", "store_id", "state_id"}
_PRICE = {"price", "price_rel_max52", "price_change", "price_rel_dept"}


def feature_label(name: str) -> tuple[str, str]:
    """(readable label, group) for a LightGBM feature name as MLForecast emits it."""
    if m := re.fullmatch(r"lag(\d+)", name):
        n = int(m.group(1))
        return (f"Sales {n} week{'s' if n > 1 else ''} ago", "Recent sales")
    if m := re.fullmatch(r"rolling_mean_lag(\d+)_window_size(\d+)", name):
        return (f"Mean sales, last {m.group(2)} weeks", "Recent sales")
    if name == "week_of_year":
        return ("Week of year", "Calendar")
    if name in _STATIC:
        return (COLUMNS[name].split(",")[0].split(":")[0], "Store / product")
    if name in _PRICE:
        return (COLUMNS[name].split(";")[0], "Price")
    return (COLUMNS.get(name, name), "Events")


def column_profile(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in df.columns:
        s = df[col]
        entry = {
            "name": col,
            "type": _kind(s),
            "meaning": COLUMNS.get(col, ""),
            "missing_share": float(s.isna().mean()),
            "distinct": int(s.nunique()),
        }
        if entry["type"] in ("number", "integer", "date") and s.notna().any():
            lo, hi = s.min(), s.max()
            entry["min"], entry["max"] = _show(lo), _show(hi)
        rows.append(entry)
    return rows


def zero_runs(panel: pd.DataFrame, min_len: int = LONG_ZERO_RUN) -> pd.DataFrame:
    """Runs of at least `min_len` consecutive zero weeks: unique_id, start, length, to_end."""
    df = panel[["unique_id", "week", "y"]].sort_values(["unique_id", "week"], kind="stable")
    zero = (df["y"].to_numpy() == 0)
    uid = df["unique_id"].to_numpy()
    new_run = np.ones(len(df), dtype=bool)
    new_run[1:] = (zero[1:] != zero[:-1]) | (uid[1:] != uid[:-1])
    run_id = np.cumsum(new_run)
    runs = pd.DataFrame({"run": run_id[zero], "unique_id": uid[zero], "week": df["week"].to_numpy()[zero]})
    agg = runs.groupby("run").agg(unique_id=("unique_id", "first"), start=("week", "min"), end=("week", "max"))
    agg["length"] = agg["end"] - agg["start"] + 1
    agg["to_end"] = agg["end"] == int(panel["week"].max())
    return agg[agg["length"] >= min_len].reset_index(drop=True)[["unique_id", "start", "length", "to_end"]]


RAW_FILES = {
    "sales_train_evaluation.csv": ("Daily unit sales, one row per item × store, one column per day", "Weekly sales (panel)"),
    "sell_prices.csv": ("Weekly shelf price per store × item × Walmart week", "price + price features (exog)"),
    "calendar.csv": ("One row per day: Walmart week, events, SNAP days per state", "Calendar (weekly sums and flags)"),
}


def raw_file_rows(stats: dict[str, tuple[int, int]]) -> list[dict]:
    """stats: file name -> (rows, columns) of the raw CSV."""
    return [
        {"file": name, "rows": rows, "columns": cols, "holds": RAW_FILES[name][0], "becomes": RAW_FILES[name][1]}
        for name, (rows, cols) in stats.items()
    ]


def listed_price_share(panel_keys: pd.DataFrame, listed: pd.DataFrame) -> tuple[int, int]:
    """(rows with a listed price, rows whose price was filled from a neighbouring week).

    panel_keys: unique_id, week; listed: unique_id, week of every price row in sell_prices.
    """
    marked = panel_keys.merge(listed.drop_duplicates().assign(_listed=True), on=["unique_id", "week"], how="left")
    n_listed = int(marked["_listed"].fillna(False).astype(bool).sum())
    return n_listed, int(len(panel_keys) - n_listed)


def training_matrix(train: pd.DataFrame, cutoff: int, train_weeks: int, n_features: int) -> dict:
    """Size of the LightGBM training set at a cutoff (same row rule as models_lgbm)."""
    rows = train[train["week"] > cutoff - train_weeks]
    first = train.groupby("unique_id")["week"].min()
    age = rows["week"].to_numpy() - rows["unique_id"].map(first).to_numpy()
    return {
        "rows": len(rows),
        "series": int(rows["unique_id"].nunique()),
        "features": int(n_features),
        "no_lag52_share": float((age < 52).mean()) if len(rows) else 0.0,
    }


def data_profile(
    panel: pd.DataFrame,
    static: pd.DataFrame,
    calendar: pd.DataFrame,
    exog: pd.DataFrame,
    raw_days: int,
    segment: pd.Series | None = None,
    raw: dict | None = None,
) -> dict:
    """Everything the Data tab shows. `segment` (unique_id -> segment) adds per-segment zero shares.

    `raw` (optional) carries facts that need the raw files: {"files": [...], "price_filled_rows": int}.
    """
    n_weeks = int(calendar["week"].max() + 1)
    first_week = panel.groupby("unique_id")["week"].min()
    ds = dict(zip(calendar["week"], calendar["ds"].dt.strftime("%Y-%m-%d")))
    runs = zero_runs(panel)
    start_year = first_week.map(lambda w: ds[int(w)][:4]).value_counts().sort_index()

    zero = panel["y"] == 0
    zero_by_segment = {}
    if segment is not None:
        seg = panel["unique_id"].map(segment)
        zero_by_segment = {str(k): float(v) for k, v in zero.groupby(seg).mean().items()}

    tables = {"panel": panel, "static": static, "calendar": calendar, "exog": exog}
    return {
        "overview": {
            "rows": len(panel),
            "series": int(panel["unique_id"].nunique()),
            "weeks": n_weeks,
            "first_week": ds[0],
            "last_week": ds[n_weeks - 1],
            "items": int(static["item_id"].nunique()),
            "stores": int(static["store_id"].nunique()),
            "departments": int(static["dept_id"].nunique()),
            "states": int(static["state_id"].nunique()),
        },
        "tables": [
            {"id": k, "title": TABLES[k][0], "grain": TABLES[k][1], "rows": len(df), "columns": column_profile(df)}
            for k, df in tables.items()
        ],
        "gaps": {
            "never_sold": int(static["unique_id"].nunique() - panel["unique_id"].nunique()),
            "late_start_series": int((first_week > 0).sum()),
            "leading_weeks_trimmed": int(first_week.sum()),
            "start_year": {str(k): int(v) for k, v in start_year.items()},
            "zero_share": float(zero.mean()),
            "zero_share_by_segment": zero_by_segment,
            "long_run_weeks": LONG_ZERO_RUN,
            "long_run_series": int(runs["unique_id"].nunique()),
            "long_run_week_share": float(runs["length"].sum() / len(panel)),
            "long_run_longest": int(runs["length"].max()) if len(runs) else 0,
            "long_run_to_end_series": int(runs.loc[runs["to_end"], "unique_id"].nunique()),
            "tail_days_dropped": int(raw_days - n_weeks * 7),
            "price_filled_rows": (raw or {}).get("price_filled_rows"),
        },
        "raw_files": (raw or {}).get("files", []),
    }


def importance_rows(imp: pd.DataFrame) -> list[dict]:
    out = []
    for r in imp.itertuples(index=False):
        label, group = feature_label(r.feature)
        out.append({"feature": r.feature, "label": label, "group": group,
                    "gain": round(float(r.gain_share), 5), "split": round(float(r.split_share), 5)})
    return out


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
        return "text"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    return "number"


def _show(v) -> str | float:
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return round(float(v), 4)
