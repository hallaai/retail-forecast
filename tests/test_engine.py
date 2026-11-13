"""Engine test on deterministic synthetic data: windows -> forecasts -> calibration -> scores."""

import numpy as np
import pandas as pd
import pytest

from m5demo import pipeline, splits
from m5demo.forecast_frame import standardize


def _synthetic_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(40):
        start = 0 if k < 34 else 100  # last 6 series are young -> "new" at later cutoffs
        level = rng.choice([0.3, 3.0, 30.0])
        for w in range(start, 140):
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * w / 52)
            rows.append({"unique_id": f"S{k:02d}", "week": w, "y": float(rng.poisson(level * seasonal))})
    df = pd.DataFrame(rows)
    df["ds"] = pd.Timestamp("2011-01-29") + pd.to_timedelta(df["week"] * 7, unit="D")
    return df


def _mean_runner(ctx: pipeline.WindowContext) -> pd.DataFrame:
    """Trailing 8-week mean, flat over the horizon (a stand-in for a real model)."""
    w = ctx.window
    level = ctx.train[ctx.train["week"] > w.cutoff - 8].groupby("unique_id")["y"].mean()
    rows = [{"unique_id": u, "week": wk, "f": v} for u, v in level.items() for wk in range(w.first_week, w.last_week + 1)]
    return standardize(pd.DataFrame(rows))


@pytest.fixture()
def run(tmp_path):
    panel = _synthetic_panel()
    windows = splits.make_windows(int(panel["week"].max()), 12, 4)
    contexts = [pipeline.build_context(panel, w) for w in windows]
    store = pipeline.ForecastStore(tmp_path, "synthetic")
    pipeline.run_forecasts(contexts, store, {"tsb": _mean_runner})
    frames = pipeline.calibrated_forecasts(contexts, store, ("tsb",))
    return contexts, store, frames, pipeline.score_all(contexts, frames)


def test_every_actual_gets_a_calibrated_forecast(run):
    contexts, _, frames, _ = run
    for ctx in contexts:
        df = frames[("tsb", ctx.window.name)]
        assert len(df) == len(ctx.actuals)
        assert df[["q10", "q50", "q90"]].notna().all().all()
        assert (df["q10"] <= df["q90"]).all() and (df["q10"] >= 0).all()


def test_segment_volumes_add_up_to_total(run):
    _, _, _, scores = run
    for _, part in scores.groupby("window"):
        total = part.loc[part["segment"] == "all", "volume"].item()
        assert part.loc[part["segment"] != "all", "volume"].sum() == pytest.approx(total)


def test_scores_are_finite_and_in_range(run):
    _, _, _, scores = run
    assert scores["wape"].between(0, 5).all()
    assert scores["coverage80"].between(0, 1).all()


def test_cache_skips_recompute(run, tmp_path):
    contexts, store, _, _ = run
    calls = []
    pipeline.run_forecasts(contexts, store, {"tsb": lambda c: calls.append(1)})
    assert calls == []
    assert set(store.runtimes()["tsb"]) == {c.window.name for c in contexts}


def test_missing_forecast_rows_raise(run):
    contexts, store, _, _ = run
    ctx = contexts[-1]
    partial = store.load("tsb", ctx.window.name).iloc[:-1]
    with pytest.raises(ValueError, match="no forecast"):
        pipeline.evaluate.scored_frame(ctx.actuals, partial, ctx.segment, ctx.window.cutoff)
