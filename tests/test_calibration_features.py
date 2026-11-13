import numpy as np
import pandas as pd
import pytest

from m5demo import calibration, features, pipeline


def test_ratio_round_trip_is_identity():
    train = pd.DataFrame({"unique_id": ["a"] * 4 + ["b"] * 4, "week": [0, 1, 2, 3] * 2, "y": [2, 4, 6, 8, 0, 0, 1, 0]})
    scale = features.trailing_mean_scale(train, cutoff=3, window=2, eps=0.1)
    assert scale["a"] == pytest.approx(7.1)
    assert scale["b"] == pytest.approx(0.6)
    ratio = features.to_ratio(train, scale)
    back = features.from_ratio(ratio.rename(columns={"y": "f"}), scale, ["f"])
    np.testing.assert_allclose(back["f"], train["y"], rtol=1e-12)


def _backtest(rng: np.random.Generator, n: int = 4000) -> pd.DataFrame:
    f = rng.uniform(1, 20, n)
    scale = f.copy()
    y = np.maximum(0, f + scale * rng.normal(0, 0.5, n))
    return pd.DataFrame({"segment": "middle", "step": 1, "y": y, "f": f, "scale": scale})


def test_calibrated_quantiles_are_ordered_non_negative_and_cover_about_80pct():
    rng = np.random.default_rng(0)
    table = calibration.residual_quantiles(_backtest(rng))
    fresh = _backtest(rng)
    out = calibration.apply_quantiles(fresh.drop(columns="y"), table)
    assert (out["q10"] <= out["q50"]).all() and (out["q50"] <= out["q90"]).all()
    assert (out["q10"] >= 0).all()
    covered = ((fresh["y"] >= out["q10"]) & (fresh["y"] <= out["q90"])).mean()
    assert 0.75 < covered < 0.85


def test_unknown_segment_step_falls_back_to_point():
    table = pd.DataFrame({"q10": [-1.0], "q50": [0.0], "q90": [1.0]}, index=pd.MultiIndex.from_tuples([("hit", 1)]))
    fc = pd.DataFrame({"segment": ["new"], "step": [1], "f": [5.0], "scale": [2.0]})
    out = calibration.apply_quantiles(fc, table)
    assert out[["q10", "q50", "q90"]].iloc[0].tolist() == [5.0, 5.0, 5.0]


def test_ensemble_is_elementwise_mean():
    a = pd.DataFrame({"unique_id": ["x"], "week": [1], "f": [2.0], "q10": [1.0], "q50": [2.0], "q90": [3.0]})
    b = a.assign(f=4.0, q10=3.0, q50=4.0, q90=5.0)
    out = pipeline._ensemble([a, b])
    assert out[["f", "q10", "q50", "q90"]].iloc[0].tolist() == [3.0, 2.0, 3.0, 4.0]
