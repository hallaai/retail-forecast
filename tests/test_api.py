"""Integration tests through the real FastAPI pipeline, with an in-memory store (no data files)."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import app as server

UID = "FOODS_3_090_CA_3"


def _fake_store():
    weeks = list(range(260, 277))
    fc = pd.DataFrame(
        [
            {"unique_id": UID, "week": w, "f": 5.0, "q10": 2.0, "q50": 5.0, "q90": 8.0, "segment": "hit", "method": m,
             "y": 4.0, "step": w - 264, "scale": 4.0}
            for m in ("naive", "lgbm_ratio")
            for w in range(265, 277)
        ]
    ).set_index("unique_id")
    panel = pd.DataFrame({"unique_id": UID, "week": weeks, "y": [4.0] * len(weeks)}).set_index("unique_id")
    ds = {w: f"2016-{w:03d}" for w in weeks}
    return fc, panel, {"cutoff": 264, "ds": ds}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server, "_store", _fake_store)
    return TestClient(server.app)


def test_series_payload_shape_and_no_internals(client):
    res = client.get(f"/api/series/{UID}")
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"id", "segment", "history_weeks", "history", "test_weeks", "actual", "forecast"}
    assert len(body["actual"]) == 12 and len(body["history"]) == 5
    assert set(body["forecast"]) == {"naive", "lgbm_ratio"}
    assert set(body["forecast"]["naive"]) == {"f", "q10", "q90", "wape"}
    assert body["forecast"]["naive"]["wape"] == pytest.approx(0.25)
    assert "scale" not in res.text and "step" not in res.text


@pytest.mark.parametrize("bad", ["x", "foods_3_090_ca_3", "A" * 41, "FOODS;DROP"])
def test_invalid_ids_are_rejected(client, bad):
    assert client.get(f"/api/series/{bad}").status_code in (400, 404)
    assert client.get(f"/api/series/{bad}").status_code != 200


def test_unknown_series_is_404(client):
    assert client.get("/api/series/HOBBIES_9_999_XX_9").status_code == 404


def test_search_validates_and_limits(client):
    assert client.get("/api/series", params={"q": "FOODS;"}).status_code == 400
    assert client.get("/api/series", params={"limit": 500}).status_code == 422
    items = client.get("/api/series", params={"q": "FOODS_3"}).json()["items"]
    assert items == [{"id": UID, "segment": "hit"}]


def test_index_is_served_with_document_skeleton(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.text.startswith("<!doctype html>") and "<title>M5 Forecast Bench</title>" in res.text
