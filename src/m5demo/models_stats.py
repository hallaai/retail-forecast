"""Local statistical baselines via StatsForecast. One call per method so each is timed on its own."""

from __future__ import annotations

import logging

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import TSB, AutoETS, CrostonSBA, Naive, SeasonalNaive

from . import config
from .forecast_frame import ds_to_week, standardize

log = logging.getLogger(__name__)

PROBABILISTIC = ("naive", "seasonal_naive", "auto_ets")  # native Gaussian intervals
POINT_ONLY = ("croston_sba", "tsb")  # quantiles come from calibration
METHODS = (*PROBABILISTIC, *POINT_ONLY)


def _model(name: str):
    factories = {
        "naive": lambda: Naive(alias=name),
        "seasonal_naive": lambda: SeasonalNaive(season_length=config.SEASON, alias=name),
        # Non-seasonal on purpose: m=52 makes AutoETS ~12x slower (0.32 s vs 0.027 s per series,
        # measured) and fits 52 seasonal states from ~5 cycles. Yearly seasonality is covered by
        # SeasonalNaive and by LightGBM's lag-52 feature.
        "auto_ets": lambda: AutoETS(season_length=1, alias=name),
        "croston_sba": lambda: CrostonSBA(alias=name),
        "tsb": lambda: TSB(alpha_d=0.1, alpha_p=0.1, alias=name),
    }
    return factories[name]()


def forecast(train: pd.DataFrame, calendar: pd.DataFrame, horizon: int, name: str) -> pd.DataFrame:
    if name not in METHODS:
        raise ValueError(f"unknown method {name}")
    frame = _run(train, calendar, horizon, name)
    if name == "seasonal_naive":
        frame = _fallback_to_naive(frame, _run(train, calendar, horizon, "naive"))
    return standardize(frame)


def _run(train: pd.DataFrame, calendar: pd.DataFrame, horizon: int, name: str) -> pd.DataFrame:
    level = [config.INTERVAL_LEVEL] if name in PROBABILISTIC else None
    sf = StatsForecast(models=[_model(name)], freq=config.FREQ, n_jobs=-1, fallback_model=Naive())
    fc = sf.forecast(df=train[["unique_id", "ds", "y"]], h=horizon, level=level)
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    frame = pd.DataFrame({"unique_id": fc["unique_id"], "week": ds_to_week(fc["ds"], calendar), "f": fc[name]})
    if level:
        lv = config.INTERVAL_LEVEL
        frame = frame.assign(q10=fc[f"{name}-lo-{lv}"], q50=fc[name], q90=fc[f"{name}-hi-{lv}"])
    return frame.sort_values(["unique_id", "week"]).reset_index(drop=True)


def _fallback_to_naive(sn: pd.DataFrame, naive: pd.DataFrame) -> pd.DataFrame:
    """SeasonalNaive is NaN for series shorter than one season: use Naive there."""
    short = sn["f"].isna().to_numpy()
    cols = ["f", "q10", "q50", "q90"]
    out = sn.copy()
    out.loc[short, cols] = naive.loc[short, cols].to_numpy()
    log.info("seasonal_naive: %d series-weeks fell back to naive", int(short.sum()))
    return out
