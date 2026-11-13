"""Write the frontend's JSON: aggregated results plus a small, fixed sample of series.

The browser never receives the raw dataset; it gets scores, weekly totals and ~100 example series.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, metrics

METHODS = {
    "naive": ("Naive", "baseline", "Last week repeated. Gaussian interval."),
    "seasonal_naive": ("Seasonal naive (52)", "baseline", "Same week last year (falls back to naive for <52 weeks)."),
    "auto_ets": ("AutoETS", "baseline", "Exponential smoothing, model picked by AICc per series (non-seasonal)."),
    "croston_sba": ("Croston SBA", "intermittent", "Size / interval smoothing for intermittent demand. Point only."),
    "tsb": ("TSB", "intermittent", "Croston variant that decays demand probability for dying items. Point only."),
    "lgbm_level": ("LightGBM, level target", "ml", "One global model, Tweedie loss, target = units."),
    "lgbm_ratio": ("LightGBM, ratio target", "ml", "Same model, target = units / trailing 26-week mean."),
    "chronos_bolt": ("Chronos-Bolt (zero-shot)", "foundation", "Pretrained model, no training, history only. Point = mean of its 9 deciles."),
    "ensemble": ("Ensemble", "ensemble", "Mean of the two candidates with the lowest backtest WAPE."),
}
# Scored, shown in side tables only (not in the main table, totals or explorer).
EXTRA_METHODS = {
    "lgbm_level_no_exog": "LightGBM level, no price / calendar features",
    "lgbm_level_price_frozen": "LightGBM level, prices frozen at cutoff",
    "chronos_bolt_median": "Chronos-Bolt, median as point",
}
CALIBRATED = {"croston_sba", "tsb", "lgbm_level", "lgbm_ratio"}
PER_SEGMENT_SAMPLE = 25
HISTORY_WEEKS = 104


def write_web_data(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    contexts: list,
    frames: dict,
    scores: pd.DataFrame,
    runtimes: dict,
    out_dir: Path,
    ensemble_parts: tuple[str, ...] = (),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    test = next(c for c in contexts if c.window.is_test)
    ds = dict(zip(calendar["week"], calendar["ds"].dt.strftime("%Y-%m-%d")))
    labels = {k: v[0] for k, v in METHODS.items()}
    methods = []
    for k, (label, family, note) in METHODS.items():
        if k == "ensemble" and ensemble_parts:
            note = f"Mean of {' and '.join(labels[p] for p in ensemble_parts)}, the pair with the lowest backtest WAPE."
        methods.append({"id": k, "label": label, "family": family, "note": note, "calibrated": k in CALIBRATED})
    summary = {
        "generated": date.today().isoformat(),
        "meta": _meta(panel, ds, contexts),
        "methods": methods,
        "extra_methods": [{"id": k, "label": v} for k, v in EXTRA_METHODS.items()],
        "ensemble_parts": list(ensemble_parts),
        "scores": _clean(scores.to_dict(orient="records")),
        "runtimes": runtimes,
        "totals": _totals(panel, test, frames, ds),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, separators=(",", ":")))
    (out_dir / "explorer.json").write_text(json.dumps(_explorer(panel, test, frames, ds), separators=(",", ":")))


def write_profile(data_profile: dict, importance: list[dict], importance_window: dict, out_dir: Path) -> None:
    """profile.json for the Data tab: table/column profile, gaps, LightGBM feature importance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": date.today().isoformat(),
        **data_profile,
        "importance": {"model": "lgbm_level", "window": importance_window, "features": importance},
    }
    (out_dir / "profile.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))


def _meta(panel: pd.DataFrame, ds: dict, contexts: list) -> dict:
    test = next(c for c in contexts if c.window.is_test)
    seg_counts = test.segment.value_counts()
    vol = test.train[test.train["week"] > test.window.cutoff - 52].groupby("unique_id")["y"].sum()
    seg_vol = vol.groupby(test.segment).sum()
    return {
        "n_series": int(panel["unique_id"].nunique()),
        "n_weeks": int(panel["week"].max() + 1),
        "first_week": ds[int(panel["week"].min())],
        "last_week": ds[int(panel["week"].max())],
        "horizon": config.HORIZON,
        "windows": [
            {"name": c.window.name, "cutoff": ds[c.window.cutoff], "from": ds[c.window.first_week], "to": ds[c.window.last_week]}
            for c in contexts
        ],
        "segments": {
            s: {"series": int(seg_counts.get(s, 0)), "volume_share": float(seg_vol.get(s, 0) / seg_vol.sum())}
            for s in config.SEGMENTS
        },
        "segment_rules": {
            "new_max_weeks": config.SEGMENT_RULES.new_max_weeks,
            "sparse_zero_share": config.SEGMENT_RULES.sparse_zero_share,
            "hit_top_share": config.SEGMENT_RULES.hit_top_share,
        },
    }


def _totals(panel: pd.DataFrame, test, frames: dict, ds: dict) -> dict:
    """Weekly totals per segment: actuals over the last 52 history weeks + test, forecasts for test."""
    hist = panel[panel["week"].between(test.window.cutoff - 51, test.window.last_week)]
    hist = hist.assign(segment=hist["unique_id"].map(test.segment)).dropna(subset=["segment"])
    out = {}
    for seg in ("all", *config.SEGMENTS):
        part = hist if seg == "all" else hist[hist["segment"] == seg]
        actual = part.groupby("week")["y"].sum()
        entry = {"weeks": [ds[int(w)] for w in actual.index], "actual": _r(actual.to_numpy()), "forecast": {}}
        for method in METHODS:
            df = frames[(method, "test")]
            df = df if seg == "all" else df[df["segment"] == seg]
            agg = df.groupby("week")[["f", "q10", "q90"]].sum()
            entry["forecast"][method] = {"f": _r(agg["f"].to_numpy()), "q10": _r(agg["q10"].to_numpy()), "q90": _r(agg["q90"].to_numpy())}
        out[seg] = entry
    return out


def _explorer(panel: pd.DataFrame, test, frames: dict, ds: dict) -> dict:
    rng = np.random.default_rng(7)
    picked = []
    for seg in config.SEGMENTS:
        ids = np.sort(test.segment[test.segment == seg].index.to_numpy())
        picked += list(rng.choice(ids, size=min(PER_SEGMENT_SAMPLE, len(ids)), replace=False))
    hist = panel[panel["unique_id"].isin(picked) & (panel["week"] > test.window.cutoff - HISTORY_WEEKS)]
    by_method = {m: frames[(m, "test")].set_index("unique_id") for m in METHODS}
    series = []
    for uid in picked:
        h = hist[hist["unique_id"] == uid].sort_values("week")
        past, fut = h[h["week"] <= test.window.cutoff], h[h["week"] > test.window.cutoff]
        fc = {}
        for m, df in by_method.items():
            rows = df.loc[[uid]].sort_values("week")
            fc[m] = {
                "f": _r(rows["f"].to_numpy()),
                "q10": _r(rows["q10"].to_numpy()),
                "q90": _r(rows["q90"].to_numpy()),
                "wape": _num(metrics.wape(fut["y"].to_numpy(), rows["f"].to_numpy())),
            }
        series.append(
            {
                "id": uid,
                "segment": test.segment[uid],
                "history_weeks": [ds[int(w)] for w in past["week"]],
                "history": _r(past["y"].to_numpy()),
                "test_weeks": [ds[int(w)] for w in fut["week"]],
                "actual": _r(fut["y"].to_numpy()),
                "forecast": fc,
            }
        )
    return {"series": series}


def _r(a: np.ndarray) -> list:
    return [None if not np.isfinite(x) else round(float(x), 2) for x in a]


def _num(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), 6)


def _clean(rows: list[dict]) -> list[dict]:
    return [{k: (_num(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]
