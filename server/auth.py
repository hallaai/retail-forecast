"""Shared-password gate (HTTP Basic) for the forecast demo server.

The browser shows its own sign-in box once per session; credentials then ride in the
``Authorization`` header of every request. That is safe only over HTTPS, so the app must
never be exposed on plain HTTP (Azure Container Apps terminates TLS at its ingress).

Configuration (all from the environment, nothing baked into the image):
  APP_PASSWORD           the shared password. Required when AUTH_REQUIRED=1.
  APP_USER               the username the box expects (default "analyst").
  AUTH_REQUIRED          "1" to fail closed: the app refuses to start without APP_PASSWORD.
  AUTH_MAX_FAILURES      wrong attempts from one address before lockout (default 10).
  AUTH_LOCKOUT_SECONDS   how long that address is locked out (default 300).

When AUTH_REQUIRED is unset/0 and no APP_PASSWORD is given, the gate is a no-op, so local
runs and the test suite are unaffected.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass

from fastapi import Request, Response, status

log = logging.getLogger("server.auth")

_OPEN_PATHS = frozenset({"/healthz"})


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


@dataclass(frozen=True)
class AuthConfig:
    """Immutable auth settings resolved from the environment at startup."""

    user: str
    password: str | None
    required: bool
    max_failures: int
    lockout_seconds: int

    @classmethod
    def from_env(cls) -> AuthConfig:
        return cls(
            user=os.environ.get("APP_USER", "analyst").strip() or "analyst",
            password=(os.environ.get("APP_PASSWORD") or "").strip() or None,
            required=os.environ.get("AUTH_REQUIRED", "").strip() in {"1", "true", "True"},
            max_failures=_int_env("AUTH_MAX_FAILURES", 10),
            lockout_seconds=_int_env("AUTH_LOCKOUT_SECONDS", 300),
        )

    @property
    def enabled(self) -> bool:
        return self.password is not None


class _Lockout:
    """Per-address failure counter with a sliding window. Process-local (one replica)."""

    def __init__(self, max_failures: int, window_seconds: int) -> None:
        self._max = max_failures
        self._window = window_seconds
        self._fails: dict[str, list[float]] = defaultdict(list)

    def locked(self, addr: str, now: float) -> bool:
        hits = [t for t in self._fails.get(addr, ()) if now - t < self._window]
        if hits:
            self._fails[addr] = hits
        else:
            self._fails.pop(addr, None)
        return len(hits) >= self._max

    def record_failure(self, addr: str, now: float) -> None:
        self._fails[addr].append(now)

    def reset(self, addr: str) -> None:
        self._fails.pop(addr, None)


def _parse_basic(header: str | None) -> tuple[str, str] | None:
    """Return (user, password) from an ``Authorization: Basic …`` header, or None."""
    if not header or not header.lower().startswith("basic "):
        return None
    import base64
    import binascii

    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    user, _, password = decoded.partition(":")
    if not _:  # no colon at all
        return None
    return user, password


def _client_addr(request: Request) -> str:
    """Caller IP, trusting the platform ingress's X-Forwarded-For first hop."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _challenge() -> Response:
    # Middleware must *return* a Response; a raised HTTPException here would bypass
    # FastAPI's handlers and surface as a 500.
    return Response(
        "Authentication required.",
        status_code=status.HTTP_401_UNAUTHORIZED,
        headers={"WWW-Authenticate": 'Basic realm="M5 forecast demo"'},
    )


def install_auth(app) -> AuthConfig:
    """Resolve config, fail closed if required, and attach the Basic-auth middleware.

    Returns the resolved config (handy for tests and logging). Idempotent per app.
    """
    config = AuthConfig.from_env()

    if config.required and not config.enabled:
        raise RuntimeError(
            "AUTH_REQUIRED=1 but APP_PASSWORD is not set. Refusing to start open. "
            "Set APP_PASSWORD (12+ chars) or clear AUTH_REQUIRED for local use."
        )

    if not config.enabled:
        log.info("Auth gate disabled (no APP_PASSWORD).")
        return config

    if len(config.password or "") < 12:
        log.warning("APP_PASSWORD is shorter than 12 characters; use a longer secret.")

    lockout = _Lockout(config.max_failures, config.lockout_seconds)
    log.info("Auth gate enabled for user %r (lockout %d/%ds).",
             config.user, config.max_failures, config.lockout_seconds)

    @app.middleware("http")
    async def _gate(request: Request, call_next):
        if request.url.path in _OPEN_PATHS:
            return await call_next(request)

        addr = _client_addr(request)
        now = time.monotonic()
        if lockout.locked(addr, now):
            return Response(
                "Too many failed attempts. Try again later.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(config.lockout_seconds)},
            )

        creds = _parse_basic(request.headers.get("authorization"))
        if creds is None:
            return _challenge()

        user_ok = secrets.compare_digest(creds[0], config.user)
        pass_ok = secrets.compare_digest(creds[1], config.password or "")
        if not (user_ok and pass_ok):
            lockout.record_failure(addr, now)
            return _challenge()

        lockout.reset(addr)
        return await call_next(request)

    return config
