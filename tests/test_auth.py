"""Auth gate tests: no-op when unset, Basic challenge/accept, lockout, fail-closed start."""

import base64
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.auth import AuthConfig, install_auth

USER = "analyst"
PW = "correct-horse-battery"


def _basic(user: str, password: str) -> dict:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/api/thing")
    def thing() -> dict:
        return {"value": 1}

    install_auth(app)
    return app


@pytest.fixture()
def guarded(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", PW)
    monkeypatch.setenv("APP_USER", USER)
    monkeypatch.setenv("AUTH_REQUIRED", "1")
    monkeypatch.setenv("AUTH_MAX_FAILURES", "3")
    monkeypatch.setenv("AUTH_LOCKOUT_SECONDS", "300")
    return TestClient(_app())


def test_disabled_when_no_password(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    client = TestClient(_app())
    assert client.get("/api/thing").status_code == 200


def test_fail_closed_when_required_without_password(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.setenv("AUTH_REQUIRED", "1")
    with pytest.raises(RuntimeError, match="APP_PASSWORD"):
        install_auth(FastAPI())


def test_healthz_is_open(guarded):
    res = guarded.get("/healthz")
    assert res.status_code == 200 and res.json() == {"ok": True}


def test_missing_credentials_challenge(guarded):
    res = guarded.get("/api/thing")
    assert res.status_code == 401
    assert res.headers.get("www-authenticate", "").lower().startswith("basic")


def test_correct_credentials_accepted(guarded):
    res = guarded.get("/api/thing", headers=_basic(USER, PW))
    assert res.status_code == 200 and res.json() == {"value": 1}


@pytest.mark.parametrize("user,pw", [(USER, "wrong"), ("intruder", PW), ("x", "y")])
def test_wrong_credentials_rejected(guarded, user, pw):
    assert guarded.get("/api/thing", headers=_basic(user, pw)).status_code == 401


def test_lockout_after_repeated_failures(guarded):
    for _ in range(3):
        assert guarded.get("/api/thing", headers=_basic(USER, "wrong")).status_code == 401
    # 4th attempt (even with the right password) is locked out for this address
    locked = guarded.get("/api/thing", headers=_basic(USER, PW))
    assert locked.status_code == 429
    assert "retry-after" in {k.lower() for k in locked.headers}


def test_config_from_env_defaults(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("APP_USER", raising=False)
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    cfg = AuthConfig.from_env()
    assert cfg.user == "analyst" and cfg.password is None
    assert cfg.required is False and cfg.enabled is False
