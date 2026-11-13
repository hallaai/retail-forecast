"""Rolling-origin splits. Pure functions, no I/O."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Window:
    name: str  # "bt1".."btN" (oldest first) or "test"
    cutoff: int  # last week index visible to the model
    horizon: int

    @property
    def first_week(self) -> int:
        return self.cutoff + 1

    @property
    def last_week(self) -> int:
        return self.cutoff + self.horizon

    @property
    def is_test(self) -> bool:
        return self.name == "test"


def make_windows(last_week: int, horizon: int, n_backtest: int) -> tuple[Window, ...]:
    """Test = last `horizon` weeks; backtest windows are the non-overlapping blocks before it."""
    if horizon <= 0 or n_backtest < 0:
        raise ValueError("horizon must be > 0 and n_backtest >= 0")
    test_cutoff = last_week - horizon
    backtests = tuple(
        Window(f"bt{i + 1}", test_cutoff - (n_backtest - i) * horizon, horizon) for i in range(n_backtest)
    )
    if backtests and backtests[0].cutoff < 0:
        raise ValueError("not enough history for the requested windows")
    return (*backtests, Window("test", test_cutoff, horizon))


def train_part(panel: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Rows the model may see. Series with no sale up to the cutoff are absent (cold start is out of scope)."""
    return panel[panel["week"] <= window.cutoff]


def actuals_part(panel: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Actuals for series that exist at the cutoff (i.e. have at least one training row)."""
    ids = panel.loc[panel["week"] <= window.cutoff, "unique_id"].unique()
    mask = panel["week"].between(window.first_week, window.last_week) & panel["unique_id"].isin(ids)
    return panel.loc[mask, ["unique_id", "week", "ds", "y"]]
