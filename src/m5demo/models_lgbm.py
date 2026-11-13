"""Global LightGBM via MLForecast, in two target variants.

* lgbm_level: target = weekly units
* lgbm_ratio: target = units / trailing 26-week mean at the cutoff (scale-free across series)

Recursive multi-step (MLForecast default): lag-1/2/4 for later horizons use earlier predictions.
Features are computed on the full history; the model trains on the last TRAIN_WEEKS weeks only.
LightGBM gives a point forecast; quantiles are added later by `calibration` from backtest residuals.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

from . import config, features
from .forecast_frame import standardize

log = logging.getLogger(__name__)
VARIANTS = ("lgbm_level", "lgbm_ratio")
# Ablations of lgbm_level: what the exogenous features are worth.
ABLATIONS = ("lgbm_level_no_exog", "lgbm_level_price_frozen")
TRAIN_WEEKS = 156


def _mlf() -> MLForecast:
    return MLForecast(
        models={"lgbm": lgb.LGBMRegressor(**config.LGBM_PARAMS)},
        freq=config.FREQ,
        lags=[1, 2, 4, 52],
        lag_transforms={1: [RollingMean(4), RollingMean(13), RollingMean(52)]},
        date_features=[features.week_of_year],
        num_threads=8,
    )


def forecast(train: pd.DataFrame, exog: pd.DataFrame, cutoff: int, horizon: int, variant: str) -> pd.DataFrame:
    mlf, ex, scale = _fit(train, exog, cutoff, variant)
    future = ex[ex["week"].between(cutoff + 1, cutoff + horizon)].drop(columns=features.STATIC_COLS)
    origin = _origin(train)
    future = future.assign(ds=origin + pd.to_timedelta(future["week"].astype(int) * 7, unit="D")).drop(columns="week")
    has_exog = len(future.columns.difference(["unique_id", "ds"])) > 0
    pred = mlf.predict(h=horizon, X_df=future if has_exog else None)

    out = pred.assign(week=((pred["ds"] - origin).dt.days // 7).astype(int), f=pred["lgbm"])
    if scale is not None:
        out = features.from_ratio(out, scale, ["f"])
    return standardize(out)


def feature_importance(train: pd.DataFrame, exog: pd.DataFrame, cutoff: int, variant: str = "lgbm_level") -> pd.DataFrame:
    """Gain and split importance of the model `forecast` would train at this cutoff.

    Columns: feature, gain_share, split_share (each sums to 1 over features), sorted by gain.
    """
    mlf, _, _ = _fit(train, exog, cutoff, variant)
    booster = mlf.models_["lgbm"].booster_
    gain = booster.feature_importance("gain").astype("float64")
    split = booster.feature_importance("split").astype("float64")
    out = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain_share": gain / gain.sum() if gain.sum() > 0 else gain,
            "split_share": split / split.sum() if split.sum() > 0 else split,
        }
    )
    return out.sort_values("gain_share", ascending=False, kind="stable").reset_index(drop=True)


def _fit(train: pd.DataFrame, exog: pd.DataFrame, cutoff: int, variant: str):
    """Build features and fit; returns (fitted MLForecast, exog for these series, ratio scale or None)."""
    if variant not in (*VARIANTS, *ABLATIONS):
        raise ValueError(f"unknown variant {variant}")
    df = train[["unique_id", "ds", "week", "y"]]
    scale = None
    if variant == "lgbm_ratio":
        scale = features.trailing_mean_scale(df, cutoff, config.RATIO_WINDOW, config.RATIO_EPS)
        df = features.to_ratio(df, scale)

    ex = exog[exog["unique_id"].isin(df["unique_id"].unique())]
    if variant == "lgbm_level_no_exog":
        ex = ex.drop(columns=features.EXOG_COLS)
    elif variant == "lgbm_level_price_frozen":
        ex = features.freeze_prices(ex, cutoff)
    fit_df = df.merge(ex, on=["unique_id", "week"])
    recent = (fit_df["week"] > cutoff - TRAIN_WEEKS).to_numpy()
    fit_df = fit_df.drop(columns="week")

    mlf = _mlf()
    X = mlf.preprocess(fit_df, static_features=features.STATIC_COLS, dropna=False)
    # preprocess keeps row count and order (y has no NaN), so the mask lines up; NaN lags stay.
    if len(X) != len(recent):
        raise RuntimeError("MLForecast.preprocess changed the row count; training-row mask would misalign")
    X = X.loc[recent]
    mlf.fit_models(X.drop(columns=["unique_id", "ds", "y"]), X["y"])
    log.info("%s cutoff=%d trained on %d rows", variant, cutoff, len(X))
    return mlf, ex, scale


def _origin(train: pd.DataFrame) -> pd.Timestamp:
    """Date of week 0, so week index <-> ds conversion works past the training range."""
    first = train.loc[train["week"].idxmin()]
    return first["ds"] - pd.Timedelta(weeks=int(first["week"]))
