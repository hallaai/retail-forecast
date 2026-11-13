import numpy as np
import pandas as pd
import pytest

from m5demo import metrics


def test_wape_known_answer():
    y = np.array([10.0, 0.0, 5.0])
    f = np.array([8.0, 1.0, 5.0])
    assert metrics.wape(y, f) == pytest.approx(3 / 15)


def test_wape_handles_zero_actual_rows_and_all_zero_total():
    assert metrics.wape(np.array([0.0, 4.0]), np.array([2.0, 4.0])) == pytest.approx(0.5)
    assert np.isnan(metrics.wape(np.zeros(3), np.ones(3)))


def test_bias_sign_over_forecast_is_positive():
    y = np.array([10.0, 10.0])
    assert metrics.bias(y, np.array([12.0, 12.0])) == pytest.approx(0.2)
    assert metrics.bias(y, np.array([8.0, 8.0])) == pytest.approx(-0.2)


def test_pinball_asymmetry():
    y = np.array([10.0])
    # Under-forecast by 2 at q=0.9 costs 0.9*2, over-forecast by 2 costs 0.1*2
    assert metrics.pinball(y, np.array([8.0]), 0.9)[0] == pytest.approx(1.8)
    assert metrics.pinball(y, np.array([12.0]), 0.9)[0] == pytest.approx(0.2)


def test_scaled_pinball_median_only_equals_wape():
    y = np.array([3.0, 0.0, 7.0])
    f = np.array([2.0, 1.0, 9.0])
    assert metrics.scaled_pinball(y, {0.5: f}) == pytest.approx(metrics.wape(y, f))


def test_coverage():
    y = np.array([1.0, 5.0, 10.0, 20.0])
    assert metrics.coverage(y, np.full(4, 2.0), np.full(4, 12.0)) == pytest.approx(0.5)


def test_mase_scale_and_exclusion_of_constant_series():
    train = pd.DataFrame({"unique_id": ["a"] * 4 + ["b"] * 3, "week": [0, 1, 2, 3, 0, 1, 2], "y": [1, 3, 1, 3, 2, 2, 2]})
    scale = metrics.naive_scale(train)
    assert scale["a"] == pytest.approx(2.0)
    assert np.isnan(scale["b"])  # constant history -> excluded
    frame = pd.DataFrame({"unique_id": ["a", "a", "b"], "y": [2.0, 4.0, 2.0], "f": [3.0, 3.0, 9.0]})
    assert metrics.mase(frame, "f", scale) == pytest.approx(0.5)
