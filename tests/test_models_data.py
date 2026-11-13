"""Adapters and I/O helpers on tiny synthetic inputs (fast; no M5 files needed)."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from m5demo import config, data, export, features, models_lgbm, models_stats, pipeline, splits
from tests.test_engine import _mean_runner, _synthetic_panel


@pytest.fixture(scope="module")
def panel():
    return _synthetic_panel(seed=1)


@pytest.fixture(scope="module")
def calendar(panel):
    weeks = pd.DataFrame({"week": range(int(panel["week"].max()) + 1)})
    weeks["ds"] = pd.Timestamp("2011-01-29") + pd.to_timedelta(weeks["week"] * 7, unit="D")
    for c in ("snap_CA", "snap_TX", "snap_WI", *features.CALENDAR_COLS):
        weeks[c] = 0
    weeks.loc[weeks["week"] % 52 == 47, "is_christmas"] = 1
    return weeks


@pytest.fixture(scope="module")
def static(panel):
    ids = panel["unique_id"].unique()
    return pd.DataFrame(
        {"unique_id": ids, "item_id": ids, "dept_id": "FOODS_1", "cat_id": "FOODS",
         "store_id": np.where(np.arange(len(ids)) % 2, "CA_1", "TX_1"), "state_id": np.where(np.arange(len(ids)) % 2, "CA", "TX")}
    )


def test_weekly_sales_sums_blocks_of_seven():
    sales = pd.DataFrame({f"d_{i + 1}": [1.0, float(i)] for i in range(16)})
    out = data.weekly_sales(sales, n_weeks=2)
    np.testing.assert_allclose(out, [[7, 7], [21, 70]])


def test_complete_weeks_drops_partial_tail():
    cal = pd.DataFrame({"date": pd.date_range("2011-01-29", periods=16), "wm_yr_wk": [1] * 7 + [2] * 7 + [3] * 2})
    cal["d_idx"] = np.arange(16)
    weeks = data.complete_weeks(cal, n_days=16)
    assert weeks["wm_yr_wk"].tolist() == [1, 2] and weeks["week"].tolist() == [0, 1]


def test_dense_long_starts_at_first_sale():
    y = np.array([[0, 0, 3, 0], [1, 0, 0, 2]], dtype=np.float32)
    out = data._dense_long(np.array(["a", "b"]), y, first_sale=np.array([2, 0]))
    assert out.groupby("unique_id")["week"].min().to_dict() == {"a": 2, "b": 0}
    assert len(out) == 6


@pytest.mark.parametrize("method", models_stats.METHODS)
def test_stats_methods_cover_every_series_and_horizon(panel, calendar, method):
    w = splits.Window("test", cutoff=127, horizon=12)
    train = splits.train_part(panel, w)
    fc = models_stats.forecast(train, calendar, 12, method)
    assert len(fc) == train["unique_id"].nunique() * 12
    assert fc["f"].notna().all() and (fc["f"] >= 0).all()
    assert fc["week"].between(128, 139).all()
    has_q = method in models_stats.PROBABILISTIC
    assert fc["q10"].notna().all() == has_q


@pytest.mark.parametrize("variant", models_lgbm.VARIANTS)
def test_lgbm_variants_forecast_all_series(panel, calendar, static, variant, monkeypatch):
    monkeypatch.setitem(config.LGBM_PARAMS, "n_estimators", 20)
    monkeypatch.setitem(config.LGBM_PARAMS, "min_child_samples", 5)
    exog = features.build_exog(panel.assign(price=2.0), static, calendar)
    w = splits.Window("test", cutoff=127, horizon=12)
    train = splits.train_part(panel, w)
    fc = models_lgbm.forecast(train, exog, w.cutoff, 12, variant)
    assert len(fc) == train["unique_id"].nunique() * 12
    assert fc["f"].notna().all() and (fc["f"] >= 0).all()
    # forecasts stay on the scale of each series (the ratio variant must undo its scaling)
    recent = train[train["week"] > 100].groupby("unique_id")["y"].mean()
    ratio = fc.groupby("unique_id")["f"].mean() / (recent + 0.5)
    assert ratio.median() == pytest.approx(1.0, abs=0.6)


def test_lgbm_feature_importance_covers_model_features(panel, calendar, static, monkeypatch):
    monkeypatch.setitem(config.LGBM_PARAMS, "n_estimators", 20)
    monkeypatch.setitem(config.LGBM_PARAMS, "min_child_samples", 5)
    exog = features.build_exog(panel.assign(price=2.0), static, calendar)
    train = splits.train_part(panel, splits.Window("test", cutoff=127, horizon=12))
    imp = models_lgbm.feature_importance(train, exog, 127)
    names = set(imp["feature"])
    assert {"lag1", "lag52", "rolling_mean_lag1_window_size13", "week_of_year", "store_id", "price"} <= names
    assert imp["gain_share"].sum() == pytest.approx(1.0) and imp["split_share"].sum() == pytest.approx(1.0)
    assert (imp["gain_share"].diff().dropna() <= 0).all()  # sorted by gain, largest first
    assert imp["gain_share"].iloc[0] > 0


def test_exog_has_one_row_per_series_week(panel, calendar, static):
    exog = features.build_exog(panel.assign(price=np.where(panel["week"] % 10 == 0, 1.5, 2.0)), static, calendar)
    assert len(exog) == len(panel)
    assert exog["price_rel_max52"].between(0, 1).all()
    assert set(features.EXOG_COLS) <= set(exog.columns)


@pytest.mark.skipif(not os.environ.get("RUN_CHRONOS"), reason="loads a ~200 MB model; set RUN_CHRONOS=1 to run")
def test_chronos_forecast_shape(panel):
    from m5demo import models_chronos

    w = splits.Window("test", cutoff=127, horizon=12)
    fc = models_chronos.forecast(splits.train_part(panel, w), w.cutoff, 12)
    assert fc[["f", "q10", "q50", "q90"]].notna().all().all()


def test_export_writes_frontend_json(panel, calendar, tmp_path):
    windows = splits.make_windows(int(panel["week"].max()), 12, 4)
    contexts = [pipeline.build_context(panel, w) for w in windows]
    store = pipeline.ForecastStore(tmp_path, "x")
    pipeline.run_forecasts(contexts, store, {m: _mean_runner for m in pipeline.ALL_METHODS})
    frames = pipeline.calibrated_forecasts(contexts, store, pipeline.ALL_METHODS)
    frames, parts = pipeline.with_ensemble(contexts, frames)
    frames = pipeline.with_median_point(contexts, frames, "chronos_bolt")
    scores = pipeline.score_all(contexts, frames)
    export.write_web_data(panel, calendar, contexts, frames, scores, store.runtimes(), tmp_path / "web", parts)
    summary = json.loads((tmp_path / "web" / "summary.json").read_text())
    explorer = json.loads((tmp_path / "web" / "explorer.json").read_text())
    assert {m["id"] for m in summary["methods"]} == set(export.METHODS)
    assert len(summary["ensemble_parts"]) == 2
    assert {"chronos_bolt_median", "lgbm_level_no_exog"} <= {r["method"] for r in summary["scores"]}
    assert summary["meta"]["n_series"] == panel["unique_id"].nunique()
    assert len(summary["totals"]["all"]["forecast"]["ensemble"]["f"]) == 12
    s = explorer["series"][0]
    assert len(s["actual"]) == 12 and set(s["forecast"]) == set(export.METHODS)
    assert "NaN" not in (tmp_path / "web" / "summary.json").read_text()
