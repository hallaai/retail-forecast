"""End-to-end run: prepare -> forecast (cached) -> calibrate + score -> export web JSON.

Usage:
    python scripts/run_pipeline.py                 # full run (all 30,490 series)
    python scripts/run_pipeline.py --sample 2000   # quick dev run on a stratified-free random sample
    python scripts/run_pipeline.py --stages score,export
    python scripts/run_pipeline.py --stages profile   # Data tab only (trains LightGBM once, then cached)
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from m5demo import config, data, export, features, models_chronos, models_lgbm, models_stats, pipeline, profile, splits

log = logging.getLogger("run")


def main() -> None:
    args = _args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore")
    stages = set(args.stages.split(","))

    if "prepare" in stages or not (config.PROCESSED_DIR / "exog.parquet").exists():
        data.save_processed()
        panel, static, calendar = data.load_processed()
        features.build_exog(panel, static, calendar).to_parquet(config.PROCESSED_DIR / "exog.parquet", index=False)

    panel, static, calendar = data.load_processed()
    tag = "full"
    if args.sample:
        ids = panel["unique_id"].drop_duplicates().sample(args.sample, random_state=0)
        panel = panel[panel["unique_id"].isin(ids)]
        tag = f"sample{args.sample}"
    windows = splits.make_windows(int(panel["week"].max()), config.HORIZON, config.N_BACKTEST_WINDOWS)
    contexts = [pipeline.build_context(panel, w) for w in windows]
    store = pipeline.ForecastStore(config.FORECAST_DIR, tag)

    if "forecast" in stages:
        exog = pd.read_parquet(config.PROCESSED_DIR / "exog.parquet")
        exog = exog[exog["unique_id"].isin(panel["unique_id"].unique())]
        h = config.HORIZON
        runners = {
            **{m: _stats_runner(m, calendar, h) for m in models_stats.METHODS},
            **{v: _lgbm_runner(v, exog, h) for v in (*models_lgbm.VARIANTS, *models_lgbm.ABLATIONS)},
            "chronos_bolt": lambda c: models_chronos.forecast(c.train, c.window.cutoff, h),
        }
        models_chronos.warm_up()  # model load (~80 s first time) is not part of the timed forecast
        pipeline.run_forecasts(contexts, store, runners)

    if "score" in stages or "export" in stages:
        frames = pipeline.calibrated_forecasts(contexts, store, pipeline.ALL_METHODS)
        frames, ensemble_parts = pipeline.with_ensemble(contexts, frames)
        frames = pipeline.with_median_point(contexts, frames, "chronos_bolt")
        scores = pipeline.score_all(contexts, frames)
        out = config.RESULTS_DIR / tag
        out.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out / "scores.csv", index=False)
        test_fc = pd.concat([df.assign(method=m) for (m, w), df in frames.items() if w == "test"])
        test_fc.to_parquet(out / "test_forecasts.parquet", index=False)
        log.info("scores written to %s", out / "scores.csv")
        if "export" in stages:
            export.write_web_data(panel, calendar, contexts, frames, scores, store.runtimes(), config.WEB_DATA_DIR, ensemble_parts)

    if "profile" in stages:
        _write_profile(panel, static, calendar, contexts, tag)


def _write_profile(panel: pd.DataFrame, static: pd.DataFrame, calendar: pd.DataFrame, contexts: list, tag: str) -> None:
    """Data tab: data profile plus LightGBM (level) feature importance at the test cutoff (cached)."""
    exog = pd.read_parquet(config.PROCESSED_DIR / "exog.parquet")
    exog = exog[exog["unique_id"].isin(panel["unique_id"].unique())]
    static = static[static["unique_id"].isin(panel["unique_id"].unique())] if tag != "full" else static
    test = next(c for c in contexts if c.window.is_test)
    raw_days = sum(c.startswith("d_") for c in pd.read_csv(config.RAW_DIR / "sales_train_evaluation.csv", nrows=0).columns)
    raw = _raw_facts(panel)
    prof = profile.data_profile(panel, static, calendar, exog, raw_days, test.segment, raw)

    cache = config.RESULTS_DIR / tag / "lgbm_importance.csv"
    if cache.exists():
        imp = pd.read_csv(cache)
    else:
        imp = models_lgbm.feature_importance(test.train, exog, test.window.cutoff)
        cache.parent.mkdir(parents=True, exist_ok=True)
        imp.to_csv(cache, index=False)
    ds = dict(zip(calendar["week"], calendar["ds"].dt.strftime("%Y-%m-%d")))
    window = {
        "name": "test",
        "cutoff": ds[test.window.cutoff],
        "train_weeks": models_lgbm.TRAIN_WEEKS,
        **profile.training_matrix(test.train, test.window.cutoff, models_lgbm.TRAIN_WEEKS, len(imp)),
    }
    export.write_profile(prof, profile.importance_rows(imp), window, config.WEB_DATA_DIR)
    log.info("profile written (%d features)", len(imp))


def _raw_facts(panel: pd.DataFrame) -> dict:
    """Facts that need the raw CSVs: their size, and how many panel rows had no listed price."""
    sales_cols = pd.read_csv(config.RAW_DIR / "sales_train_evaluation.csv", nrows=0).columns
    with open(config.RAW_DIR / "sales_train_evaluation.csv", encoding="utf-8") as fh:
        sales_rows = sum(1 for _ in fh) - 1  # minus the header
    cal_raw = data.load_calendar(config.RAW_DIR)
    prices = pd.read_csv(config.RAW_DIR / "sell_prices.csv")
    weeks = data.complete_weeks(cal_raw, sum(c.startswith("d_") for c in sales_cols))
    listed = prices.assign(unique_id=prices["item_id"] + "_" + prices["store_id"]).merge(weeks[["wm_yr_wk", "week"]], on="wm_yr_wk")
    _, filled = profile.listed_price_share(panel[["unique_id", "week"]], listed[["unique_id", "week"]])
    files = profile.raw_file_rows(
        {
            "sales_train_evaluation.csv": (sales_rows, len(sales_cols)),
            "sell_prices.csv": (len(prices), prices.shape[1]),
            "calendar.csv": (len(cal_raw), cal_raw.shape[1] - 1),  # minus the d_idx helper column
        }
    )
    return {"files": files, "price_filled_rows": filled}


def _stats_runner(method: str, calendar: pd.DataFrame, h: int):
    return lambda c: models_stats.forecast(c.train, calendar, h, method)


def _lgbm_runner(variant: str, exog: pd.DataFrame, h: int):
    return lambda c: models_lgbm.forecast(c.train, exog, c.window.cutoff, h, variant)


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--stages", default="forecast,score,export,profile")
    return ap.parse_args()


if __name__ == "__main__":
    main()
