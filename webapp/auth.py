"""Two fixed accounts (admin, operator), signed-cookie sessions, CSRF tokens and login throttling.

Credentials are static: they come from .env (CONSOLE_ADMIN_USER / CONSOLE_ADMIN_PASSWORD and
CONSOLE_OPERATOR_USER / CONSOLE_OPERATOR_PASSWORD). When a password is not set, a strong random one is
generated on first start, stored in instance/credentials.json (mode 600) and printed once to the console.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

from .store import INSTANCE_DIR

log = logging.getLogger(__name__)
CREDENTIALS_FILE = INSTANCE_DIR / "credentials.json"
SECRET_FILE = INSTANCE_DIR / "session_secret"
ROLES = ("admin", "operator")

MAX_FAILURES = 5
LOCKOUT_SECONDS = 60


@dataclass(frozen=True)
class User:
    username: str
    role: str  # admin | operator

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class Accounts:
    def __init__(self) -> None:
        self._accounts = self._load()
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _load(self) -> dict[str, tuple[str, str]]:
        """username -> (role, sha256(password))."""
        stored = {}
        if CREDENTIALS_FILE.exists():
            stored = json.loads(CREDENTIALS_FILE.read_text())
        generated = {}
        accounts = {}
        for role in ROLES:
            user = os.getenv(f"CONSOLE_{role.upper()}_USER") or stored.get(role, {}).get("username") or role
            password = os.getenv(f"CONSOLE_{role.upper()}_PASSWORD") or stored.get(role, {}).get("password")
            if not password:
                password = secrets.token_urlsafe(12)
                generated[role] = {"username": user, "password": password}
            accounts[user] = (role, _digest(password))
        if generated:
            INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
            CREDENTIALS_FILE.write_text(json.dumps({**stored, **generated}, indent=1))
            os.chmod(CREDENTIALS_FILE, 0o600)
            lines = "\n".join(f"  {r:<9} username: {c['username']:<10} password: {c['password']}" for r, c in generated.items())
            print(f"\nGenerated console credentials (saved to {CREDENTIALS_FILE.relative_to(INSTANCE_DIR.parent)}):\n{lines}\n", flush=True)
        if len(accounts) != len(ROLES):
            raise RuntimeError("Admin and operator must have different usernames")
        return accounts

    def usernames(self) -> dict[str, str]:
        return {role: user for user, (role, _) in self._accounts.items()}

    def authenticate(self, username: str, password: str, client: str) -> tuple[User | None, str]:
        key = f"{client}:{username.lower()}"
        with self._lock:
            recent = [t for t in self._failures.get(key, []) if time.monotonic() - t < LOCKOUT_SECONDS]
            self._failures[key] = recent
            if len(recent) >= MAX_FAILURES:
                wait = int(LOCKOUT_SECONDS - (time.monotonic() - recent[0])) + 1
                return None, f"Too many attempts. Try again in {wait} seconds."
        account = self._accounts.get(username)
        # Compare against a dummy digest for unknown users so timing does not reveal valid usernames.
        expected = account[1] if account else _digest(secrets.token_hex(8))
        if account and hmac.compare_digest(expected, _digest(password)):
            with self._lock:
                self._failures.pop(key, None)
            return User(username, account[0]), ""
        with self._lock:
            self._failures.setdefault(key, []).append(time.monotonic())
        log.info("Failed console login for %r from %s", username, client)
        return None, "Incorrect username or password."


def _digest(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def session_secret() -> str:
    if env := os.getenv("CONSOLE_SECRET_KEY"):
        return env
    if SECRET_FILE.exists():
        return SECRET_FILE.read_text().strip()
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    value = secrets.token_urlsafe(48)
    SECRET_FILE.write_text(value)
    os.chmod(SECRET_FILE, 0o600)
    return value


# --- request helpers ----------------------------------------------------------------------


class LoginRequired(Exception):
    pass


def current_user(request: Request) -> User | None:
    data = request.session.get("user")
    if not data or data.get("role") not in ROLES:
        return None
    return User(data["username"], data["role"])


def require_user(request: Request) -> User:
    user = current_user(request)
    if user is None:
        raise LoginRequired()
    return user


def require_admin(request: Request) -> User:
    user = require_user(request)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Only an admin can do this.")
    return user


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(24)
        request.session["csrf"] = token
    return token


async def verify_csrf(request: Request) -> None:
    expected = request.session.get("csrf")
    sent = request.headers.get("x-csrf-token")
    if sent is None and request.headers.get("content-type", "").startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        sent = (await request.form()).get("csrf")
    if not expected or not sent or not hmac.compare_digest(str(sent), expected):
        raise HTTPException(status_code=403, detail="Your session expired. Reload the page and try again.")
