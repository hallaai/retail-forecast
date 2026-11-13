"""Local demo server: serves the frontend and looks up any series' test-window forecasts.

Run:  .venv/Scripts/python -m uvicorn server.app:app --port 8050
Everything heavy was computed by scripts/run_pipeline.py; this only reads its outputs.
"""

from __future__ import annotations

import logging
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5demo import config

log = logging.getLogger("server")
ID_PATTERN = re.compile(r"^[A-Z0-9_]{3,40}$")
HISTORY_WEEKS = 104
RUN_TAG = "full"


@lru_cache(maxsize=1)
def _store() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Loaded once; read-only afterwards."""
    fc = pd.read_parquet(config.RESULTS_DIR / RUN_TAG / "test_forecasts.parquet")
    cutoff = int(fc["week"].min()) - 1
    panel = pd.read_parquet(config.PROCESSED_DIR / "panel.parquet", columns=["unique_id", "week", "y"])
    panel = panel[panel["week"] > cutoff - HISTORY_WEEKS]
    cal = pd.read_parquet(config.PROCESSED_DIR / "calendar.parquet", columns=["week", "ds"])
    ds = dict(zip(cal["week"], cal["ds"].dt.strftime("%Y-%m-%d")))
    return fc.set_index("unique_id").sort_index(), panel.set_index("unique_id").sort_index(), {"cutoff": cutoff, "ds": ds}


app = FastAPI(title="M5 forecast demo", docs_url=None, redoc_url=None)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/series")
def search(q: str = Query("", max_length=40), limit: int = Query(20, ge=1, le=50)) -> dict:
    fc, _, _ = _store()
    needle = q.strip().upper()
    if needle and not re.fullmatch(r"[A-Z0-9_]+", needle):
        raise HTTPException(400, "Use letters, digits and underscores only, e.g. FOODS_3_090_CA_3.")
    seg = fc.loc[fc["method"] == "naive", "segment"]
    seg = seg[~seg.index.duplicated()]
    hits = seg[seg.index.str.contains(needle, regex=False)] if needle else seg
    return {"items": [{"id": i, "segment": s} for i, s in hits.head(limit).items()]}


@app.get("/api/series/{uid}")
def series(uid: str) -> dict:
    if not ID_PATTERN.fullmatch(uid):
        raise HTTPException(400, "Series ids look like FOODS_3_090_CA_3.")
    fc, panel, meta = _store()
    if uid not in fc.index:
        raise HTTPException(404, "No series with that id exists at the test cutoff.")
    rows = fc.loc[[uid]]
    hist = panel.loc[[uid]].sort_values("week") if uid in panel.index else panel.iloc[:0]
    past, fut = hist[hist["week"] <= meta["cutoff"]], hist[hist["week"] > meta["cutoff"]]
    return {
        "id": uid,
        "segment": rows["segment"].iloc[0],
        "history_weeks": [meta["ds"][int(w)] for w in past["week"]],
        "history": _r(past["y"]),
        "test_weeks": [meta["ds"][int(w)] for w in fut["week"]],
        "actual": _r(fut["y"]),
        "forecast": {m: _method_payload(g, fut["y"]) for m, g in rows.groupby("method")},
    }


def _method_payload(g: pd.DataFrame, actual: pd.Series) -> dict:
    g = g.sort_values("week")
    denom = float(actual.sum())
    wape = float((actual.to_numpy() - g["f"].to_numpy()).__abs__().sum() / denom) if denom > 0 else None
    return {"f": _r(g["f"]), "q10": _r(g["q10"]), "q90": _r(g["q90"]), "wape": None if wape is None else round(wape, 4)}


def _r(s: pd.Series) -> list:
    return [None if pd.isna(x) else round(float(x), 2) for x in s]


@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def index() -> str:
    """web/index.html is written as a fragment (the artifact host adds the skeleton); add it here too."""
    body = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8">{body}</html>'


app.mount("/", StaticFiles(directory=ROOT / "web"), name="web")
