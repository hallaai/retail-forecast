import itertools

import numpy as np
import pandas as pd
import pytest

from m5demo import segments, splits
from m5demo.config import SegmentRules


def _panel(series: dict[str, list[float]], start: dict[str, int] | None = None) -> pd.DataFrame:
    rows = []
    for uid, ys in series.items():
        s = (start or {}).get(uid, 0)
        rows += [{"unique_id": uid, "week": s + i, "y": y} for i, y in enumerate(ys)]
    df = pd.DataFrame(rows)
    df["ds"] = pd.Timestamp("2011-01-29") + pd.to_timedelta(df["week"] * 7, unit="D")
    return df


def test_windows_are_contiguous_non_overlapping_and_end_at_last_week():
    ws = splits.make_windows(last_week=276, horizon=12, n_backtest=4)
    assert [w.name for w in ws] == ["bt1", "bt2", "bt3", "bt4", "test"]
    assert ws[-1].last_week == 276
    for a, b in itertools.pairwise(ws):
        assert b.first_week == a.last_week + 1


def test_windows_reject_too_short_history():
    with pytest.raises(ValueError):
        splits.make_windows(last_week=30, horizon=12, n_backtest=4)


def test_train_never_contains_forecast_weeks_and_actuals_skip_unborn_series():
    panel = _panel({"old": [1.0] * 40, "late": [2.0] * 5}, start={"late": 35})
    w = splits.Window("test", cutoff=29, horizon=10)
    train = splits.train_part(panel, w)
    assert train["week"].max() == 29
    actual = splits.actuals_part(panel, w)
    assert set(actual["unique_id"]) == {"old"}  # "late" starts after the cutoff: cold start, out of scope


def test_segment_precedence_and_rules():
    rules = SegmentRules(new_max_weeks=52, sparse_zero_share=0.3, hit_top_share=0.5)
    series = {
        "new": [5.0] * 10,
        "sparse": ([0.0, 0.0, 3.0] * 30)[:80],
        "big": [100.0] * 80,
        "small": [10.0] * 80,
    }
    panel = _panel(series, start={"new": 70})
    prof = segments.series_profile(panel, cutoff=79)
    seg = segments.assign_segments(prof, rules)
    assert seg.to_dict() == {"new": "new", "sparse": "sparse", "big": "hit", "small": "middle"}


def test_segments_use_only_training_data():
    panel = _panel({"a": [0.0] * 60 + [50.0] * 20})
    prof = segments.series_profile(panel[panel["week"] <= 59], cutoff=59)
    assert prof.loc["a", "volume_52w"] == 0
    assert prof.loc["a", "zero_share_52w"] == pytest.approx(1.0)
    assert np.isfinite(prof.loc["a", "age_weeks"])
