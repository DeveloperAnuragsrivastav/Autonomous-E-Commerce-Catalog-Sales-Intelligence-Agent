"""Run configuration: CLI flags + environment, resolved once and passed through the graph."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VENDOR_SHEET = "vendor data"
SALES_SHEET = "sales data"


@dataclass(frozen=True)
class Settings:
    input_path: Path
    output_dir: Path = PROJECT_ROOT / "output"
    images_dir: Path = PROJECT_ROOT / "images"
    cache_dir: Path = PROJECT_ROOT / ".cache"

    # Providers
    google_api_key: str | None = None
    groq_api_key: str | None = None
    serpapi_api_key: str | None = None
    # Tried in order; each Gemini model has its own free-tier daily quota, Groq is the last resort.
    text_models: tuple[str, ...] = ("gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash")
    fallback_text_model: str = "openai/gpt-oss-120b"  # Groq
    image_model: str = "gemini-2.5-flash-image"
    image_provider: str = "auto"  # auto | gemini | composite

    # Scope controls (None = all)
    skus: tuple[str, ...] | None = None
    image_limit: int | None = None
    competitor_skus: tuple[str, ...] | None = None
    competitor_limit: int = 8
    competitors_per_sku: int = 3

    skip_images: bool = False
    skip_competitors: bool = False
    use_cache: bool = True
    max_workers: int = 4

    storefront_search_url: str = "https://benzara.com/search/suggest.json"

    extra: dict = field(default_factory=dict)


def load_settings(**overrides) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    env = {
        "google_api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "serpapi_api_key": os.getenv("SERPAPI_API_KEY"),
    }
    if os.getenv("TEXT_MODELS"):
        env["text_models"] = tuple(m.strip() for m in os.environ["TEXT_MODELS"].split(",") if m.strip())
    for key, env_name in [("fallback_text_model", "FALLBACK_TEXT_MODEL"), ("image_model", "IMAGE_MODEL")]:
        if os.getenv(env_name):
            env[key] = os.environ[env_name]
    merged = {**env, **{k: v for k, v in overrides.items() if v is not None}}
    return Settings(**merged)
