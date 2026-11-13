import json

import numpy as np
import pandas as pd
import pytest

from m5demo import profile


def _tables():
    weeks = np.arange(30)
    calendar = pd.DataFrame({"week": weeks, "ds": pd.date_range("2011-01-29", periods=30, freq="7D")})
    # a: sells every week. b: starts at week 5, then 15 zero weeks (8..22), then sells. c: dies at week 20.
    y_a = np.ones(30)
    y_b = np.r_[np.ones(3), np.zeros(15), np.ones(7)]
    y_c = np.r_[np.ones(20), np.zeros(10)]
    panel = pd.concat(
        [
            pd.DataFrame({"unique_id": "a", "week": weeks, "y": y_a}),
            pd.DataFrame({"unique_id": "b", "week": weeks[5:], "y": y_b}),
            pd.DataFrame({"unique_id": "c", "week": weeks, "y": y_c}),
        ],
        ignore_index=True,
    )
    panel["ds"] = panel["week"].map(dict(zip(calendar["week"], calendar["ds"])))
    panel["price"] = 1.0
    static = pd.DataFrame(
        {"unique_id": ["a", "b", "c", "d"], "item_id": ["i1", "i2", "i3", "i4"], "dept_id": "D", "cat_id": "C",
         "store_id": "S", "state_id": "CA"}
    )
    exog = panel[["unique_id", "week", "price"]].assign(price_change=0.0)
    return panel, static, calendar, exog


def test_zero_runs_finds_only_long_runs_and_flags_runs_to_the_end():
    panel, *_ = _tables()
    runs = profile.zero_runs(panel, min_len=10)
    assert sorted(runs["unique_id"]) == ["b", "c"]
    b = runs[runs["unique_id"] == "b"].iloc[0]
    c = runs[runs["unique_id"] == "c"].iloc[0]
    assert (b["start"], b["length"], b["to_end"]) == (8, 15, False)
    assert (c["start"], c["length"], c["to_end"]) == (20, 10, True)


def test_zero_runs_do_not_join_across_series():
    # c ends on zeros, the next series starts on zeros: two runs of 2, never one of 4.
    panel = pd.DataFrame({"unique_id": ["c"] * 3 + ["d"] * 3, "week": [0, 1, 2] * 2, "y": [1, 0, 0, 0, 0, 1]})
    assert profile.zero_runs(panel, min_len=3).empty


def test_data_profile_counts_and_gaps():
    panel, static, calendar, exog = _tables()
    seg = pd.Series({"a": "hit", "b": "sparse", "c": "middle"})
    p = profile.data_profile(panel, static, calendar, exog, raw_days=30 * 7 + 2, segment=seg)
    o, g = p["overview"], p["gaps"]
    assert (o["rows"], o["series"], o["weeks"]) == (85, 3, 30)
    assert g["never_sold"] == 1                       # d is in static but never sold
    assert (g["late_start_series"], g["leading_weeks_trimmed"]) == (1, 5)
    assert g["zero_share"] == pytest.approx(25 / 85)
    assert g["zero_share_by_segment"] == {"hit": 0.0, "middle": pytest.approx(10 / 30), "sparse": pytest.approx(15 / 25)}
    assert (g["long_run_series"], g["long_run_to_end_series"], g["long_run_longest"]) == (1, 0, 15)  # c's run is 10 < 13
    assert g["tail_days_dropped"] == 2
    assert sum(g["start_year"].values()) == 3
    json.dumps(p, allow_nan=False)                    # serialisable, no NaN


def test_column_profile_types_ranges_and_missing():
    df = pd.DataFrame({"y": [1.0, np.nan, 3.0], "ds": pd.to_datetime(["2011-01-29", "2011-02-05", None]),
                       "store_id": pd.Categorical(["A", "B", "A"]), "week": np.array([0, 1, 2], dtype=np.int16)})
    cols = {c["name"]: c for c in profile.column_profile(df)}
    assert cols["y"]["type"] == "number" and cols["y"]["min"] == 1.0 and cols["y"]["max"] == 3.0
    assert cols["y"]["missing_share"] == pytest.approx(1 / 3)
    assert cols["ds"]["type"] == "date" and cols["ds"]["max"] == "2011-02-05"
    assert cols["store_id"]["type"] == "text" and "min" not in cols["store_id"] and cols["store_id"]["distinct"] == 2
    assert cols["week"]["type"] == "integer"


@pytest.mark.parametrize(
    ("name", "label", "group"),
    [
        ("lag1", "Sales 1 week ago", "Recent sales"),
        ("lag52", "Sales 52 weeks ago", "Recent sales"),
        ("rolling_mean_lag1_window_size13", "Mean sales, last 13 weeks", "Recent sales"),
        ("week_of_year", "Week of year", "Calendar"),
        ("store_id", "Store", "Store / product"),
        ("price", "Shelf price", "Price"),
        ("snap_days", "SNAP days in the week in the series' state", "Events"),
        ("something_new", "something_new", "Events"),
    ],
)
def test_feature_labels(name, label, group):
    assert profile.feature_label(name) == (label, group)


def test_importance_rows_keep_order_and_shares():
    imp = pd.DataFrame({"feature": ["lag1", "price"], "gain_share": [0.7, 0.3], "split_share": [0.4, 0.6]})
    rows = profile.importance_rows(imp)
    assert [r["feature"] for r in rows] == ["lag1", "price"]
    assert sum(r["gain"] for r in rows) == pytest.approx(1.0)
    assert rows[1]["group"] == "Price"


def test_listed_price_share_counts_filled_rows():
    panel, *_ = _tables()
    keys = panel[["unique_id", "week"]]
    listed = keys[keys["week"] % 2 == 0]  # every other week has a listed price
    n_listed, n_filled = profile.listed_price_share(keys, pd.concat([listed, listed]))  # duplicates must not inflate
    assert n_listed == len(listed)
    assert n_listed + n_filled == len(keys)


def test_training_matrix_uses_the_lgbm_row_rule():
    panel, *_ = _tables()
    m = profile.training_matrix(panel, cutoff=29, train_weeks=10, n_features=25)
    assert m["rows"] == len(panel[panel["week"] > 19])
    assert m["series"] == 3 and m["features"] == 25
    assert m["no_lag52_share"] == pytest.approx(1.0)  # all synthetic series are younger than 52 weeks


def test_raw_file_rows_describe_known_files():
    rows = profile.raw_file_rows({"calendar.csv": (1969, 13)})
    assert rows == [{"file": "calendar.csv", "rows": 1969, "columns": 13,
                     "holds": profile.RAW_FILES["calendar.csv"][0], "becomes": profile.RAW_FILES["calendar.csv"][1]}]


def test_data_profile_carries_raw_facts_and_defaults_without_them():
    panel, static, calendar, exog = _tables()[:4]
    with_raw = profile.data_profile(panel, static, calendar, exog, raw_days=212, raw={"files": [{"file": "x"}], "price_filled_rows": 7})
    assert with_raw["gaps"]["price_filled_rows"] == 7 and with_raw["raw_files"] == [{"file": "x"}]
    without = profile.data_profile(panel, static, calendar, exog, raw_days=212)
    assert without["gaps"]["price_filled_rows"] is None and without["raw_files"] == []
    json.dumps(without, allow_nan=False)
