"""Connection tests for the admin console. Each uses a call that costs no generation quota."""
from __future__ import annotations

from dataclasses import dataclass

import requests

TIMEOUT = 20


@dataclass
class TestResult:
    ok: bool
    message: str
    detail: str = ""


def test_provider(provider: str, key: str | None, configured_models: list[str] | None = None) -> TestResult:
    if not key:
        return TestResult(False, "No key saved.")
    try:
        return {"google_api_key": _gemini, "groq_api_key": _groq, "serpapi_api_key": _serpapi}[provider](key, configured_models or [])
    except requests.Timeout:
        return TestResult(False, "The provider did not answer within 20 seconds.")
    except requests.RequestException as exc:
        return TestResult(False, "Could not reach the provider.", type(exc).__name__)


def _gemini(key: str, models: list[str]) -> TestResult:
    resp = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                        params={"pageSize": 200}, headers={"x-goog-api-key": key}, timeout=TIMEOUT)
    if resp.status_code in (400, 401, 403):
        return TestResult(False, "Google rejected this key.", _error(resp))
    resp.raise_for_status()
    names = {m["name"].removeprefix("models/") for m in resp.json().get("models", [])}
    missing = [m for m in models if m not in names]
    detail = f"{len(names)} models available."
    if missing:
        return TestResult(False, f"Key works, but these models are not available: {', '.join(missing)}.", detail)
    return TestResult(True, "Connected.", detail + " Text works on the free tier; image editing needs billing enabled.")


def _groq(key: str, models: list[str]) -> TestResult:
    resp = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT)
    if resp.status_code in (401, 403):
        return TestResult(False, "Groq rejected this key.", _error(resp))
    resp.raise_for_status()
    ids = {m["id"] for m in resp.json().get("data", [])}
    missing = [m for m in models if m not in ids]
    if missing:
        return TestResult(False, f"Key works, but {', '.join(missing)} is not available on it.", f"{len(ids)} models available.")
    return TestResult(True, "Connected.", f"{len(ids)} models available.")


def _serpapi(key: str, _models: list[str]) -> TestResult:
    resp = requests.get("https://serpapi.com/account.json", params={"api_key": key}, timeout=TIMEOUT)
    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if resp.status_code >= 400 or data.get("error"):
        return TestResult(False, "SerpAPI rejected this key.", data.get("error", f"HTTP {resp.status_code}"))
    left = data.get("total_searches_left", data.get("plan_searches_left"))
    return TestResult(True, "Connected.", f"{data.get('plan_name', 'Plan')}: {left} searches left this month. A full run uses about 16.")


def _error(resp: requests.Response) -> str:
    try:
        return resp.json().get("error", {}).get("message", "")[:200]
    except ValueError:
        return f"HTTP {resp.status_code}"
