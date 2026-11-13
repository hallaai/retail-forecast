"""Experiment constants. Every stage reads its settings from here.

Time conventions
----------------
* One period = one Walmart week (Saturday..Friday), labelled by its Saturday (`ds`).
* `week` is a 0-based integer index over complete weeks (week 0 starts 2011-01-29).
* A cutoff `c` is the last week index the model may see; it forecasts weeks c+1..c+H.
* The trailing 2 days of the M5 history (an incomplete week) are dropped, never fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FORECAST_DIR = ROOT / "data" / "forecasts"
RESULTS_DIR = ROOT / "data" / "results"
WEB_DATA_DIR = ROOT / "web" / "data"

FREQ = "W-SAT"
HORIZON = 12
N_BACKTEST_WINDOWS = 4
SEASON = 52
QUANTILES = (0.1, 0.5, 0.9)
INTERVAL_LEVEL = 80  # central interval -> q0.1 / q0.9


@dataclass(frozen=True)
class SegmentRules:
    """Segments are assigned per cutoff from training data only (no look-ahead)."""

    new_max_weeks: int = 52  # fewer weeks since first sale -> "new"
    sparse_zero_share: float = 0.30  # share of zero weeks in the last 52 -> "sparse"
    hit_top_share: float = 0.10  # top 10% of established series by 52w volume -> "hit"


SEGMENTS = ("hit", "middle", "sparse", "new")
SEGMENT_RULES = SegmentRules()

# Ratio-target variant: y is divided by the series' mean over the trailing window at the cutoff.
RATIO_WINDOW = 26
RATIO_EPS = 0.1

LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "n_estimators": 400,
    "learning_rate": 0.06,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "max_bin": 127,
    "num_threads": 12,
    "verbosity": -1,
    "random_state": 42,
    # deterministic=True was measured 2.5x slower on this machine; seed + fixed threads instead.
}

CHRONOS_MODEL = "amazon/chronos-bolt-small"
CHRONOS_BATCH = 512
