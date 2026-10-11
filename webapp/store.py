"""Console settings: API keys and pipeline defaults set by an admin.

Stored server-side in instance/settings.json (gitignored, mode 600). A value saved here overrides the same
value in .env, so the CLI and the console can share a machine. Secrets are never returned to the browser in
full: views get a masked form and where the value came from.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from agent.config import PROJECT_ROOT, Settings, load_settings

INSTANCE_DIR = PROJECT_ROOT / "instance"
SECRETS = {
    "google_api_key": ("Google AI Studio", "GOOGLE_API_KEY", "Gemini text generation and, with billing enabled, image editing."),
    "groq_api_key": ("Groq", "GROQ_API_KEY", "Fallback language model when every Gemini model is out of quota."),
    "serpapi_api_key": ("SerpAPI", "SERPAPI_API_KEY", "Google Shopping search for competitor benchmarking."),
}
IMAGE_PROVIDERS = {
    "auto": "Auto: Gemini image editing, local composite when unavailable",
    "gemini": "Gemini image editing only",
    "composite": "Local composite only (free)",
}
DEFAULTS = {
    "text_models": list(Settings.__dataclass_fields__["text_models"].default),
    "fallback_text_model": Settings.__dataclass_fields__["fallback_text_model"].default,
    "image_model": Settings.__dataclass_fields__["image_model"].default,
    "image_provider": "auto",
    "competitor_limit": 0,
    "competitors_per_sku": 3,
    "max_workers": 4,
}


@dataclass
class SecretView:
    key: str
    label: str
    env_name: str
    purpose: str
    masked: str
    source: str  # console | env | unset


class SettingsStore:
    def __init__(self, path: Path = INSTANCE_DIR / "settings.json") -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    # --- secrets --------------------------------------------------------------------------

    def secret(self, key: str) -> tuple[str | None, str]:
        stored = self._read().get("secrets", {}).get(key)
        if stored:
            return stored, "console"
        env_value = os.getenv(SECRETS[key][1])
        return (env_value, "env") if env_value else (None, "unset")

    def secret_views(self) -> list[SecretView]:
        load_settings(input_path=PROJECT_ROOT)  # loads .env into the environment
        views = []
        for key, (label, env_name, purpose) in SECRETS.items():
            value, source = self.secret(key)
            masked = f"{'•' * 8}{value[-4:]}" if value and len(value) > 8 else ("•" * 8 if value else "")
            views.append(SecretView(key, label, env_name, purpose, masked, source))
        return views

    def set_secret(self, key: str, value: str | None) -> None:
        if key not in SECRETS:
            raise KeyError(key)
        with self._lock:
            data = self._read()
            secrets = data.setdefault("secrets", {})
            if value:
                secrets[key] = value.strip()
            else:
                secrets.pop(key, None)
            self._write(data)

    # --- pipeline defaults ----------------------------------------------------------------

    def pipeline(self) -> dict:
        return {**DEFAULTS, **self._read().get("pipeline", {})}

    def set_pipeline(self, values: dict) -> None:
        with self._lock:
            data = self._read()
            data["pipeline"] = {**data.get("pipeline", {}), **{k: v for k, v in values.items() if k in DEFAULTS}}
            self._write(data)

    # --- resolved settings for a run ------------------------------------------------------

    def run_settings(self, **overrides) -> Settings:
        base = load_settings(input_path=overrides.get("input_path", PROJECT_ROOT))  # loads .env
        p = self.pipeline()
        resolved = {
            "google_api_key": self.secret("google_api_key")[0],
            "groq_api_key": self.secret("groq_api_key")[0],
            "serpapi_api_key": self.secret("serpapi_api_key")[0],
            "text_models": tuple(p["text_models"]) or base.text_models,
            "fallback_text_model": p["fallback_text_model"] or base.fallback_text_model,
            "image_model": p["image_model"] or base.image_model,
            "image_provider": p["image_provider"],
            "competitor_limit": int(p["competitor_limit"]),
            "competitors_per_sku": int(p["competitors_per_sku"]),
            "max_workers": int(p["max_workers"]),
        }
        return load_settings(**{**resolved, **overrides})

    def missing_required(self) -> list[str]:
        """Keys without which a run cannot do its core work."""
        missing = []
        if not (self.secret("google_api_key")[0] or self.secret("groq_api_key")[0]):
            missing.append("a language model key (Google AI Studio or Groq)")
        if not self.secret("serpapi_api_key")[0]:
            missing.append("SerpAPI (competitor analysis will be skipped)")
        return missing
