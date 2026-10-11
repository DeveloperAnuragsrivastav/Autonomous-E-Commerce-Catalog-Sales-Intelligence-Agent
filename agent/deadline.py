"""Hard deadlines for external calls.

SDK timeouts do not cover every failure: a connection stuck in SYN_SENT on a broken IPv6 path kept a Gemini
request blocked for 8+ minutes despite a 120 s client timeout. Running the call on a daemon thread and waiting
with a deadline guarantees the pipeline moves on; the abandoned thread finishes or dies on its own.
"""
from __future__ import annotations

import threading
from typing import Callable, TypeVar

T = TypeVar("T")


class DeadlineExceeded(TimeoutError):
    pass


def call_with_deadline(fn: Callable[[], T], seconds: float, what: str = "call") -> T:
    result: dict = {}
    done = threading.Event()

    def run() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # re-raised in the caller's thread
            result["error"] = exc
        finally:
            done.set()

    threading.Thread(target=run, name=f"deadline-{what}", daemon=True).start()
    if not done.wait(seconds):
        raise DeadlineExceeded(f"{what} did not finish within {seconds:.0f}s")
    if "error" in result:
        raise result["error"]
    return result["value"]
