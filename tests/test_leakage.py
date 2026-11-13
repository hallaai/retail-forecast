"""Regression tests from the independent review: row alignment, no use of future actuals, ensemble choice."""

import numpy as np
import pandas as pd
import pytest

from m5demo import config, features, models_lgbm, models_stats, pipeline, splits
from m5demo.forecast_frame import sanitize, standardize
from tests.test_engine import _mean_runner, _synthetic_panel
from tests.test_models_data import calendar, panel, static  # noqa: F401  (fixtures)


def test_mlforecast_preprocess_keeps_row_count_and_order(panel, static, calendar):  # noqa: F811
    exog = features.build_exog(panel.assign(price=2.0), static, calendar)
    df = panel[panel["week"] <= 127].merge(exog, on=["unique_id", "week"]).sample(frac=1.0, random_state=3)
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    X = models_lgbm._mlf(models_lgbm.SPECS["lgbm_level"]).preprocess(df.drop(columns="week"), static_features=features.STATIC_COLS, dropna=False)
    assert len(X) == len(df)
    assert (X["unique_id"].to_numpy() == df["unique_id"].to_numpy()).all()
    expected_lag1 = df.groupby("unique_id")["y"].shift(1).to_numpy()
    np.testing.assert_allclose(X["lag1"].to_numpy(), expected_lag1, equal_nan=True)


@pytest.mark.parametrize("method", ["naive", "auto_ets", "tsb"])
def test_stats_forecast_ignores_future_actuals(panel, calendar, method):  # noqa: F811
    w = splits.Window("test", cutoff=115, horizon=12)
    poisoned = panel.assign(y=np.where(panel["week"] > w.cutoff, 9999.0, panel["y"]))
    a = models_stats.forecast(splits.train_part(panel, w), calendar, 12, method)
    b = models_stats.forecast(splits.train_part(poisoned, w), calendar, 12, method)
    pd.testing.assert_frame_equal(a, b)


def test_lgbm_ignores_future_actuals(panel, static, calendar, monkeypatch):  # noqa: F811
    monkeypatch.setitem(config.LGBM_PARAMS, "n_estimators", 15)
    monkeypatch.setitem(config.LGBM_PARAMS, "num_threads", 1)
    exog = features.build_exog(panel.assign(price=2.0), static, calendar)
    w = splits.Window("test", cutoff=115, horizon=12)
    poisoned = panel.assign(y=np.where(panel["week"] > w.cutoff, 9999.0, panel["y"]))
    a = models_lgbm.forecast(splits.train_part(panel, w), exog, w.cutoff, 12, "lgbm_level")
    b = models_lgbm.forecast(splits.train_part(poisoned, w), exog, w.cutoff, 12, "lgbm_level")
    pd.testing.assert_frame_equal(a, b)


def test_freeze_prices_only_touches_future_weeks(panel, static, calendar):  # noqa: F811
    priced = panel.assign(price=1.0 + panel["week"] / 100)
    exog = features.build_exog(priced, static, calendar)
    frozen = features.freeze_prices(exog, cutoff=100)
    past = exog["week"] <= 100
    pd.testing.assert_frame_equal(frozen[past], exog[past])
    fut = frozen[~past]
    assert (fut["price"] == 2.0).all() and (fut["price_change"] == 0).all()


def test_sanitize_replaces_infinite_quantiles_with_point():
    df = pd.DataFrame({"unique_id": ["a", "a"], "week": [1, 2], "f": [3.0, 4.0], "q10": [1.0, -np.inf], "q50": [3.0, 4.0], "q90": [np.inf, 6.0]})
    out = standardize(sanitize(df))
    assert np.isfinite(out[["q10", "q50", "q90"]].to_numpy()).all()
    assert out["q90"].tolist() == [3.0, 6.0] and out["q10"].tolist() == [1.0, 4.0]


def test_ensemble_pair_is_chosen_on_backtests_only(tmp_path):
    synth = _synthetic_panel(seed=2)
    contexts = [pipeline.build_context(synth, w) for w in splits.make_windows(int(synth["week"].max()), 12, 4)]
    store = pipeline.ForecastStore(tmp_path, "e")

    def scaled(k):
        return lambda c: _mean_runner(c).assign(f=lambda d: d["f"] * k)

    pipeline.run_forecasts(contexts, store, {"lgbm_level": _mean_runner, "lgbm_ratio": scaled(3.0), "chronos_bolt": scaled(1.05)})
    frames = pipeline.calibrated_forecasts(contexts, store, pipeline.ENSEMBLE_CANDIDATES)
    # sabotage the test window of the best pair: the choice must not change
    test = next(c.window.name for c in contexts if c.window.is_test)
    frames[("lgbm_level", test)] = frames[("lgbm_level", test)].assign(f=1e6)
    _, parts = pipeline.with_ensemble(contexts, frames)
    assert set(parts) == {"lgbm_level", "chronos_bolt"}


def test_ensemble_refuses_misaligned_members():
    a = pd.DataFrame({"unique_id": ["x", "y"], "week": [1, 1], "f": [1.0, 2.0], "q10": 0.0, "q50": 1.0, "q90": 2.0})
    with pytest.raises(ValueError, match="row-aligned"):
        pipeline._ensemble([a, a.iloc[::-1].reset_index(drop=True)])
