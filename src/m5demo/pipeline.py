"""Orchestration: run every method on every window, cache forecasts, calibrate, score.

Forecasts are cached at data/forecasts/<method>/<window>.parquet, keyed by the set of series
(sample size) so a dev run on a sample never mixes with a full run.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration, config, evaluate, features, metrics, segments, splits
from .forecast_frame import sanitize

log = logging.getLogger(__name__)

STATS = ("naive", "seasonal_naive", "auto_ets", "croston_sba", "tsb")
LGBM = ("lgbm_level", "lgbm_ratio")
CHRONOS = ("chronos_bolt",)
ABLATIONS = ("lgbm_level_no_exog", "lgbm_level_price_frozen")
ENSEMBLE = "ensemble"  # mean of the pair of candidates with the lowest backtest WAPE (point and quantiles)
ENSEMBLE_CANDIDATES = ("lgbm_level", "lgbm_ratio", "chronos_bolt")
ALL_METHODS = (*STATS, *LGBM, *CHRONOS, *ABLATIONS)
NEEDS_CALIBRATION = ("croston_sba", "tsb", "lgbm_level", "lgbm_ratio")


@dataclass(frozen=True)
class WindowContext:
    window: splits.Window
    train: pd.DataFrame
    actuals: pd.DataFrame
    segment: pd.Series  # unique_id -> segment
    mase_scale: pd.Series  # unique_id -> in-sample naive MAE
    level_scale: pd.Series  # unique_id -> trailing mean (+eps), for calibration


def build_context(panel: pd.DataFrame, window: splits.Window) -> WindowContext:
    train = splits.train_part(panel, window)
    profile = segments.series_profile(train, window.cutoff)
    return WindowContext(
        window=window,
        train=train,
        actuals=splits.actuals_part(panel, window),
        segment=segments.assign_segments(profile),
        mase_scale=metrics.naive_scale(train),
        level_scale=features.trailing_mean_scale(train, window.cutoff, config.RATIO_WINDOW, config.RATIO_EPS),
    )


class ForecastStore:
    def __init__(self, root: Path, tag: str) -> None:
        self.root = root / tag
        self.runtime_path = self.root / "runtimes.json"

    def path(self, method: str, window: str) -> Path:
        return self.root / method / f"{window}.parquet"

    def has(self, method: str, window: str) -> bool:
        return self.path(method, window).exists()

    def load(self, method: str, window: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(method, window))

    def save(self, method: str, window: str, fc: pd.DataFrame, seconds: float) -> None:
        p = self.path(method, window)
        p.parent.mkdir(parents=True, exist_ok=True)
        fc.to_parquet(p, index=False)
        runtimes = self.runtimes()
        runtimes.setdefault(method, {})[window] = round(seconds, 2)
        self.runtime_path.write_text(json.dumps(runtimes, indent=1))

    def runtimes(self) -> dict:
        return json.loads(self.runtime_path.read_text()) if self.runtime_path.exists() else {}


def run_forecasts(
    contexts: list[WindowContext],
    store: ForecastStore,
    runners: dict[str, Callable[[WindowContext], pd.DataFrame]],
) -> None:
    """runners: method -> fn(ctx) returning its forecast. Each call is timed; cached results are skipped."""
    for ctx in contexts:
        for method, fn in runners.items():
            if store.has(method, ctx.window.name):
                continue
            t0 = time.perf_counter()
            fc = fn(ctx)
            elapsed = time.perf_counter() - t0
            store.save(method, ctx.window.name, fc, elapsed)
            log.info("%s %s: %d rows in %.1fs", method, ctx.window.name, len(fc), elapsed)


def _scored(ctx: WindowContext, fc: pd.DataFrame) -> pd.DataFrame:
    df = evaluate.scored_frame(ctx.actuals, fc, ctx.segment, ctx.window.cutoff)
    return df.assign(scale=df["unique_id"].map(ctx.level_scale).to_numpy())


def calibrated_forecasts(contexts: list[WindowContext], store: ForecastStore, methods: tuple[str, ...]) -> dict:
    """{(method, window): scored frame}. Point-only methods get backtest-calibrated quantiles.

    Calibration for backtest window k uses the other backtest windows (leave-one-window-out),
    for the test window it uses all backtest windows. No window ever calibrates on itself.
    """
    frames = {(m, c.window.name): _scored(c, sanitize(store.load(m, c.window.name))) for m in methods for c in contexts}
    backtests = [c.window.name for c in contexts if not c.window.is_test]
    for m in methods:
        if m not in NEEDS_CALIBRATION:
            continue
        for ctx in contexts:
            sources = [w for w in backtests if w != ctx.window.name]
            table = calibration.residual_quantiles(pd.concat([frames[(m, w)] for w in sources]))
            frames[(m, ctx.window.name)] = calibration.apply_quantiles(frames[(m, ctx.window.name)], table)
    return frames


def with_ensemble(contexts: list[WindowContext], frames: dict) -> tuple[dict, tuple[str, str]]:
    """Add the ensemble: the candidate pair with the lowest mean backtest WAPE (test never consulted)."""
    backtests = [c.window.name for c in contexts if not c.window.is_test]
    pairs = list(itertools.combinations(ENSEMBLE_CANDIDATES, 2))

    def backtest_wape(pair: tuple[str, str]) -> float:
        scores = []
        for w in backtests:
            avg = _ensemble([frames[(m, w)] for m in pair])
            scores.append(metrics.wape(avg["y"].to_numpy(), avg["f"].to_numpy()))
        return float(np.mean(scores))

    ranked = sorted(pairs, key=backtest_wape)
    parts = ranked[0]
    log.info("ensemble members chosen on backtest WAPE: %s", parts)
    out = dict(frames)
    for ctx in contexts:
        out[(ENSEMBLE, ctx.window.name)] = _ensemble([frames[(m, ctx.window.name)] for m in parts])
    return out, parts


def with_median_point(contexts: list[WindowContext], frames: dict, method: str) -> dict:
    """Derived method `<method>_median`: same forecasts, but the median (q50) as the point."""
    out = dict(frames)
    for ctx in contexts:
        df = frames[(method, ctx.window.name)]
        out[(f"{method}_median", ctx.window.name)] = df.assign(f=df["q50"])
    return out


def _ensemble(parts: list[pd.DataFrame]) -> pd.DataFrame:
    base = parts[0].copy()
    for p in parts[1:]:
        if not (p["unique_id"].to_numpy() == base["unique_id"].to_numpy()).all() or not (p["week"].to_numpy() == base["week"].to_numpy()).all():
            raise ValueError("ensemble members are not row-aligned")
    for col in ("f", "q10", "q50", "q90"):
        base[col] = np.mean([p[col].to_numpy() for p in parts], axis=0)
    return base


def score_all(contexts: list[WindowContext], frames: dict) -> pd.DataFrame:
    by_name = {c.window.name: c for c in contexts}
    rows = []
    for (method, window), df in frames.items():
        for row in evaluate.score(df, by_name[window].mase_scale):
            rows.append({"method": method, "window": window, **row})
    return pd.DataFrame(rows)
