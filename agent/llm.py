"""Structured-output LLM client: Gemini primary, Groq fallback, retries, rate limiting, disk cache.

Every call returns a validated Pydantic object or raises `LLMError`; callers never parse raw text.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .cache import DiskCache
from .deadline import DeadlineExceeded, call_with_deadline
from .config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
REQUEST_TIMEOUT_S = 60
DEADLINE_S = REQUEST_TIMEOUT_S + 30  # hard stop even when the SDK's own timeout does not fire


class LLMError(Exception):
    pass


class _Retryable(Exception):
    """Transient failure (rate limit, 5xx, timeout, malformed JSON) worth retrying."""


class _QuotaExhausted(Exception):
    """Provider quota is used up for the day: skip straight to the fallback provider."""


class _Unreachable(Exception):
    """The call blew its hard deadline (stalled network path): stop using this model for the run."""


@dataclass
class LLMResult:
    value: BaseModel
    provider: str
    model: str
    cached: bool = False


class _RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.interval = 60.0 / per_minute
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            time.sleep(delay)


class LLM:
    def __init__(self, settings: Settings, cache: DiskCache) -> None:
        self.settings = settings
        self.cache = cache
        self._gemini = None
        self._groq = None
        self._exhausted: set[str] = set()  # models whose daily quota ran out during this run
        self._limits = {"gemini": _RateLimiter(8), "groq": _RateLimiter(6)}  # Groq free tier: 8k tokens/min
        if settings.google_api_key:
            from google import genai
            from google.genai import types

            # The SDK has no default timeout: without one a stalled response hangs the whole run.
            self._gemini = genai.Client(api_key=settings.google_api_key,
                                        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000))
        if settings.groq_api_key:
            from groq import Groq

            self._groq = Groq(api_key=settings.groq_api_key, max_retries=0, timeout=REQUEST_TIMEOUT_S)
        if not (self._gemini or self._groq):
            raise LLMError("No LLM provider configured: set GOOGLE_API_KEY and/or GROQ_API_KEY in .env")

    def structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        namespace: str,
        temperature: float = 0.4,
    ) -> LLMResult:
        key = DiskCache.key(system, prompt, schema.model_json_schema(), temperature)
        if (hit := self.cache.get(namespace, key)) is not None:
            return LLMResult(schema.model_validate(hit["value"]), hit["provider"], hit["model"], cached=True)

        errors = []
        for provider, model in self._chain():
            try:
                value = self._call(provider, model, system, prompt, schema, temperature)
            except (_QuotaExhausted, _Unreachable) as exc:
                if model not in self._exhausted:  # parallel workers can hit the limit together
                    why = "daily quota exhausted" if isinstance(exc, _QuotaExhausted) else "not responding"
                    log.warning("%s %s; falling back for the rest of the run", model, why)
                    errors.append(f"{model}: {why}")
                self._exhausted.add(model)
                continue
            except Exception as exc:  # retries exhausted or non-retryable
                errors.append(f"{model}: {_short(exc)}")
                log.warning("LLM %s failed for %s: %s", model, namespace, _short(exc))
                continue
            self.cache.set(namespace, key, {"value": value.model_dump(), "provider": provider, "model": model})
            return LLMResult(value, provider, model)
        raise LLMError("; ".join(errors) or "no provider available")

    def _chain(self) -> list[tuple[str, str]]:
        chain = [("gemini", m) for m in self.settings.text_models if self._gemini]
        if self._groq:
            chain.append(("groq", self.settings.fallback_text_model))
        return [(p, m) for p, m in chain if m not in self._exhausted]

    def _call(self, provider: str, model: str, system: str, prompt: str, schema: type[T], temperature: float):
        call = self._gemini_call if provider == "gemini" else self._groq_call

        @retry(
            retry=retry_if_exception(lambda e: isinstance(e, _Retryable)),
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=2, min=2, max=40),
            reraise=True,
        )
        def attempt():
            self._limits[provider].wait()
            return call(model, system, prompt, schema, temperature)

        return attempt()

    # --- providers -------------------------------------------------------------------------

    def _gemini_call(self, model: str, system: str, prompt: str, schema: type[T], temperature: float):
        from google.genai import errors, types

        try:
            resp = call_with_deadline(lambda: self._gemini.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            ), DEADLINE_S, f"gemini {model}")
        except DeadlineExceeded as exc:
            raise _Unreachable(model) from exc
        except errors.APIError as exc:
            msg = str(exc)
            if exc.code == 429 and "PerDay" in msg:
                raise _QuotaExhausted(model) from exc
            if exc.code == 404:
                raise _QuotaExhausted(model) from exc  # model retired: skip it for the run
            if exc.code in (429, 500, 502, 503, 504):
                raise _Retryable(msg[:200]) from exc
            raise
        except (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.TransportError) as exc:
            raise _Retryable(f"network: {_short(exc)}") from exc
        return _validate(schema, resp.parsed if isinstance(resp.parsed, schema) else resp.text)

    def _groq_call(self, model: str, system: str, prompt: str, schema: type[T], temperature: float):
        import groq

        schema_hint = json.dumps(schema.model_json_schema())
        try:
            resp = call_with_deadline(lambda: self._groq.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"{system}\n\nRespond with a single JSON object matching this JSON Schema:\n{schema_hint}"},
                    {"role": "user", "content": prompt},
                ],
            ), DEADLINE_S, f"groq {model}")
        except DeadlineExceeded as exc:
            raise _Unreachable(model) from exc
        except (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError, groq.APITimeoutError) as exc:
            raise _Retryable(_short(exc)) from exc
        return _validate(schema, resp.choices[0].message.content)


def _validate(schema: type[T], payload) -> T:
    if isinstance(payload, schema):
        return payload
    if not payload:
        raise _Retryable("empty response")
    text = re.sub(r"^```(?:json)?|```$", "", str(payload).strip()).strip()
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise _Retryable(f"schema validation failed: {exc.errors()[:2]}") from exc


def _short(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:160]}"
