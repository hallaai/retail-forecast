"""Global LightGBM via MLForecast: one model for all series, with a choice of target scaling.

Every variant is a `Spec` (target scaling, objective, row weights, exogenous features):

* lgbm_level            target = weekly units, Tweedie loss. No explicit scaling: Tweedie's log link
                        makes effects multiplicative, and recent-sales lags anchor each series' level.
* lgbm_ratio            target = units / trailing 26-week mean at the cutoff, Tweedie loss (scale-free).
* lgbm_log1p            target = log1p(units), squared-error loss; back via expm1 times a smearing factor,
                        because expm1 of a mean on the log scale under-forecasts the mean in units.
* lgbm_local_scaled     target = (units - series mean) / series std (MLForecast LocalStandardScaler),
                        squared-error loss. Must not use Tweedie: scaled targets are negative.
* lgbm_level_weighted   lgbm_level with rows weighted by the series' trailing 52-week mean volume.
* ablations             lgbm_level without exogenous features / with prices frozen at the cutoff.

Target transforms that MLForecast owns (log1p, local scaling) are applied before lags are built and
inverted inside the recursive predict, so lags and predictions stay on one scale.
Recursive multi-step (MLForecast default): lag-1/2/4 for later horizons use earlier predictions.
Features are computed on the full history; the model trains on the last TRAIN_WEEKS weeks only.
LightGBM gives a point forecast; quantiles are added later by `calibration` from backtest residuals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import GlobalSklearnTransformer, LocalStandardScaler
from sklearn.preprocessing import FunctionTransformer

from . import config, features
from .forecast_frame import standardize

log = logging.getLogger(__name__)
TRAIN_WEEKS = 156
WEIGHT_WINDOW = 52

# Objectives whose labels must be >= 0 (LightGBM refuses negative labels for them).
NON_NEGATIVE_OBJECTIVES = frozenset({"tweedie", "poisson", "gamma"})
# Targets that can be negative after transformation.
SIGNED_TARGETS = frozenset({"local_standard"})


@dataclass(frozen=True)
class Spec:
    target: str = "level"  # level | ratio | log1p | local_standard
    objective: str = "tweedie"  # tweedie | regression (squared error)
    weighted: bool = False  # weight rows by trailing 52-week mean volume
    exog: str = "all"  # all | none | frozen_prices


SPECS = {
    "lgbm_level": Spec(),
    "lgbm_ratio": Spec(target="ratio"),
    "lgbm_log1p": Spec(target="log1p", objective="regression"),
    "lgbm_local_scaled": Spec(target="local_standard", objective="regression"),
    "lgbm_level_weighted": Spec(weighted=True),
    "lgbm_level_no_exog": Spec(exog="none"),
    "lgbm_level_price_frozen": Spec(exog="frozen_prices"),
}
VARIANTS = ("lgbm_level", "lgbm_ratio")
# Scaling / weighting options, scored next to the two main variants.
SCALING = ("lgbm_log1p", "lgbm_local_scaled", "lgbm_level_weighted")
# Ablations of lgbm_level: what the exogenous features are worth.
ABLATIONS = ("lgbm_level_no_exog", "lgbm_level_price_frozen")


def validate(spec: Spec) -> None:
    """Reject combinations LightGBM would fail on, or that make no sense."""
    if spec.target not in ("level", "ratio", "log1p", "local_standard"):
        raise ValueError(f"unknown target scaling {spec.target!r}")
    if spec.objective in NON_NEGATIVE_OBJECTIVES and spec.target in SIGNED_TARGETS:
        raise ValueError(
            f"objective={spec.objective!r} needs targets >= 0, but target={spec.target!r} makes them negative "
            "(standard scaling / differencing). Use objective='regression', or a non-negative target such as "
            "'ratio' or 'log1p'."
        )


def _mlf(spec: Spec) -> MLForecast:
    params = {**config.LGBM_PARAMS, "objective": spec.objective}
    if spec.objective != "tweedie":
        params.pop("tweedie_variance_power", None)
    transforms = {
        "log1p": [GlobalSklearnTransformer(FunctionTransformer(np.log1p, np.expm1))],
        "local_standard": [LocalStandardScaler()],
    }.get(spec.target)
    return MLForecast(
        models={"lgbm": lgb.LGBMRegressor(**params)},
        freq=config.FREQ,
        lags=[1, 2, 4, 52],
        lag_transforms={1: [RollingMean(4), RollingMean(13), RollingMean(52)]},
        date_features=[features.week_of_year],
        target_transforms=transforms,
        num_threads=8,
    )


def forecast(train: pd.DataFrame, exog: pd.DataFrame, cutoff: int, horizon: int, variant: str) -> pd.DataFrame:
    fitted = _fit(train, exog, cutoff, variant)
    future = fitted.exog[fitted.exog["week"].between(cutoff + 1, cutoff + horizon)].drop(columns=features.STATIC_COLS)
    origin = _origin(train)
    future = future.assign(ds=origin + pd.to_timedelta(future["week"].astype(int) * 7, unit="D")).drop(columns="week")
    has_exog = len(future.columns.difference(["unique_id", "ds"])) > 0
    pred = fitted.mlf.predict(h=horizon, X_df=future if has_exog else None)

    out = pred.assign(week=((pred["ds"] - origin).dt.days // 7).astype(int), f=pred["lgbm"])
    if fitted.ratio_scale is not None:
        out = features.from_ratio(out, fitted.ratio_scale, ["f"])
    if fitted.smearing != 1.0:
        out = out.assign(f=(out["f"] + 1.0) * fitted.smearing - 1.0)
    return standardize(out)


def feature_importance(train: pd.DataFrame, exog: pd.DataFrame, cutoff: int, variant: str = "lgbm_level") -> pd.DataFrame:
    """Gain and split importance of the model `forecast` would train at this cutoff.

    Columns: feature, gain_share, split_share (each sums to 1 over features), sorted by gain.
    """
    booster = _fit(train, exog, cutoff, variant).mlf.models_["lgbm"].booster_
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


@dataclass(frozen=True)
class _Fitted:
    mlf: MLForecast
    exog: pd.DataFrame
    ratio_scale: pd.Series | None  # set for the ratio target
    smearing: float  # log1p target: mean of exp(residual) on the log scale; 1.0 otherwise


def _fit(train: pd.DataFrame, exog: pd.DataFrame, cutoff: int, variant: str) -> _Fitted:
    if variant not in SPECS:
        raise ValueError(f"unknown variant {variant}")
    spec = SPECS[variant]
    validate(spec)
    df = train[["unique_id", "ds", "week", "y"]]
    ratio_scale = None
    if spec.target == "ratio":
        ratio_scale = features.trailing_mean_scale(df, cutoff, config.RATIO_WINDOW, config.RATIO_EPS)
        df = features.to_ratio(df, ratio_scale)

    ex = exog[exog["unique_id"].isin(df["unique_id"].unique())]
    if spec.exog == "none":
        ex = ex.drop(columns=features.EXOG_COLS)
    elif spec.exog == "frozen_prices":
        ex = features.freeze_prices(ex, cutoff)
    fit_df = df.merge(ex, on=["unique_id", "week"])
    recent = (fit_df["week"] > cutoff - TRAIN_WEEKS).to_numpy()
    fit_df = fit_df.drop(columns="week")

    mlf = _mlf(spec)
    X = mlf.preprocess(fit_df, static_features=features.STATIC_COLS, dropna=False)
    # preprocess keeps row count and order (y has no NaN), so the mask lines up; NaN lags stay.
    if len(X) != len(recent):
        raise RuntimeError("MLForecast.preprocess changed the row count; training-row mask would misalign")
    X = X.loc[recent]
    feats = X.drop(columns=["unique_id", "ds", "y"])
    fit_kwargs = None
    if spec.weighted:
        fit_kwargs = {"lgbm": {"sample_weight": volume_weights(train, cutoff, X["unique_id"])}}
    mlf.fit_models(feats, X["y"].to_numpy(), models_fit_kwargs=fit_kwargs)
    smearing = 1.0
    if spec.target == "log1p":
        # Duan smearing on in-sample residuals (log scale); slightly optimistic, so it under-corrects a little.
        resid = X["y"].to_numpy() - mlf.models_["lgbm"].predict(feats)
        smearing = float(np.mean(np.exp(resid)))
    log.info("%s cutoff=%d trained on %d rows (smearing %.3f)", variant, cutoff, len(X), smearing)
    return _Fitted(mlf, ex, ratio_scale, smearing)


def volume_weights(train: pd.DataFrame, cutoff: int, row_ids: pd.Series) -> np.ndarray:
    """Per training row: its series' trailing 52-week mean units, normalised to mean 1.

    Gives big sellers more say than the unweighted loss does, moving training toward WAPE's view,
    where every unit sold (not every series) counts the same.
    """
    level = features.trailing_mean_scale(train, cutoff, WEIGHT_WINDOW, config.RATIO_EPS)
    w = row_ids.map(level).fillna(config.RATIO_EPS).to_numpy(dtype="float64")
    return w / w.mean()


def _origin(train: pd.DataFrame) -> pd.Timestamp:
    """Date of week 0, so week index <-> ds conversion works past the training range."""
    first = train.loc[train["week"].idxmin()]
    return first["ds"] - pd.Timedelta(weeks=int(first["week"]))
