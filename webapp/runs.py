"""Run registry and the background worker that executes the agent for the console.

Each run lives in runs/<id>/: the uploaded workbook, output/, images/, run.json (status, per-step progress,
options, who started it) and run.log. One run executes at a time; the CLI's output folder is imported as the
first run so a fresh console has results to browse without spending API quota.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from agent.cache import DiskCache
from agent.config import PROJECT_ROOT
from agent.graph import RunContext, build_graph

from .store import SettingsStore

log = logging.getLogger(__name__)

RUNS_DIR = PROJECT_ROOT / "runs"
STEPS = [
    ("load_input", "Load workbook"),
    ("analyse_products", "Analyse products"),
    ("generate_content", "Generate content"),
    ("generate_images", "Lifestyle images"),
    ("research_competitors", "Competitors"),
    ("analyse_sales", "Sales analysis"),
    ("write_outputs", "Write outputs"),
]
ACTIVE = {"queued", "running", "cancelling"}
RUN_ID_RE = re.compile(r"^[a-z0-9-]{6,40}$")


class RunBusy(Exception):
    pass


@dataclass
class RunOptions:
    skus: list[str] = field(default_factory=list)
    image_provider: str = "auto"
    image_limit: int | None = None
    skip_images: bool = False
    skip_competitors: bool = False
    competitor_limit: int = 0
    use_cache: bool = True


class RunManager:
    def __init__(self, store: SettingsStore, root: Path = RUNS_DIR) -> None:
        self.store = store
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active: str | None = None
        self._cancel: dict[str, threading.Event] = {}
        self._logs: dict[str, deque] = {}
        self._live: dict[str, dict] = {}  # in-memory state of the active run (fresher than run.json)
        self._recover_interrupted()

    # --- queries --------------------------------------------------------------------------

    def list(self) -> list[dict]:
        runs = []
        for path in self.root.glob("*/run.json"):
            try:
                meta = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            runs.append(self.get(meta["id"]) if meta["id"] in self._live else meta)
        return sorted(runs, key=lambda r: r["created_at"], reverse=True)

    def get(self, run_id: str) -> dict | None:
        if not RUN_ID_RE.match(run_id):
            return None
        if (live := self._live.get(run_id)) is not None:
            return json.loads(json.dumps(live, default=str))  # snapshot, safe to read while the worker writes
        path = self.root / run_id / "run.json"
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def dir(self, run_id: str) -> Path:
        if not RUN_ID_RE.match(run_id):
            raise ValueError("invalid run id")
        return self.root / run_id

    def log_tail(self, run_id: str, lines: int = 200) -> list[str]:
        if run_id in self._logs:
            return list(self._logs[run_id])[-lines:]
        path = self.dir(run_id) / "run.log"
        return path.read_text().splitlines()[-lines:] if path.exists() else []

    @property
    def active_run(self) -> str | None:
        return self._active

    # --- lifecycle ------------------------------------------------------------------------

    def start(self, *, source: Path, input_name: str, options: RunOptions, user: str, move: bool) -> str:
        with self._lock:
            if self._active:
                raise RunBusy(self._active)
            run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
            run_dir = self.root / run_id
            (run_dir / "input").mkdir(parents=True)
            target = run_dir / "input" / _safe_name(input_name)
            if move:
                shutil.move(str(source), target)
            else:
                shutil.copy2(source, target)
            meta = {
                "id": run_id, "input_name": target.name, "created_at": _now(), "created_by": user,
                "status": "queued", "started_at": None, "finished_at": None, "error": "",
                "options": options.__dict__, "current_step": None,
                "steps": {name: {"label": label, "status": "pending", "seconds": None, "detail": "",
                                 "done": None, "total": None, "item": ""} for name, label in STEPS},
            }
            self._save(meta)
            self._active = run_id
            self._cancel[run_id] = threading.Event()
            self._logs[run_id] = deque(maxlen=400)
        threading.Thread(target=self._execute, args=(run_id, target, options), name=f"run-{run_id}", daemon=True).start()
        return run_id

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            if self._active != run_id:
                return False
            self._cancel[run_id].set()
            meta = self._live.get(run_id) or self.get(run_id)
            meta["status"] = "cancelling"
            self._save(meta)
        return True

    def delete(self, run_id: str) -> None:
        with self._lock:
            if self._active == run_id:
                raise RunBusy(run_id)
            shutil.rmtree(self.dir(run_id), ignore_errors=True)

    # --- worker ---------------------------------------------------------------------------

    def _execute(self, run_id: str, input_path: Path, options: RunOptions) -> None:
        run_dir = self.root / run_id
        handler = _RunLogHandler(self._logs[run_id], run_dir / "run.log")
        cancel = self._cancel[run_id]
        logging.getLogger().addHandler(handler)
        meta = self.get(run_id)
        meta.update(status="running", started_at=_now())
        self._live[run_id] = meta
        self._save(meta)
        last_save = [0.0]

        def on_event(event: dict) -> None:
            step = meta["steps"].get(event.get("step"))
            if step is None:
                return
            if cancel.is_set():
                meta["status"] = "cancelling"
            if event["type"] == "step":
                # A step the graph skipped only because a cancel was already pending never ran at all.
                never_ran = event["status"] == "cancelled" and event.get("detail") == "run cancelled"
                step["status"] = "not_started" if never_ran else event["status"]
                if event["status"] == "cancelled" and not never_ran:
                    meta["stopped_at"] = event["step"]
                if event["status"] == "running":
                    meta["current_step"] = event["step"]
                else:
                    step.update(seconds=event.get("seconds"), detail=event.get("detail", ""))
            elif event["type"] == "progress":
                step.update(done=event["done"], total=event["total"], item=event.get("label", ""))
            if event["type"] == "step" or time.monotonic() - last_save[0] > 0.7:
                last_save[0] = time.monotonic()
                self._save(meta)

        try:
            settings = self.store.run_settings(
                input_path=input_path, output_dir=run_dir / "output", images_dir=run_dir / "images",
                skus=tuple(options.skus) or None, image_provider=options.image_provider,
                image_limit=options.image_limit, skip_images=options.skip_images,
                skip_competitors=options.skip_competitors or not self.store.secret("serpapi_api_key")[0],
                competitor_limit=options.competitor_limit, use_cache=options.use_cache,
            )
            ctx = RunContext(settings=settings, cache=DiskCache(settings.cache_dir, enabled=settings.use_cache),
                             on_event=on_event, cancel=cancel)
            final = build_graph(ctx).invoke({"steps": []})
            statuses = [s["status"] for s in final.get("steps", [])]
            if final.get("fatal_error"):
                meta.update(status="failed", error=final["fatal_error"])
            elif "cancelled" in statuses:
                meta["status"] = "cancelled"
            elif "failed" in statuses:
                meta["status"] = "completed_with_errors"
            else:
                meta["status"] = "succeeded"
            if meta["status"] in ("failed", "cancelled"):
                _mark_unfinished(meta)
        except Exception as exc:  # unexpected crash: record it, never leave the run "running"
            log.exception("Run %s crashed", run_id)
            meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            _mark_unfinished(meta)
        finally:
            meta.update(finished_at=_now(), current_step=None)
            self._save(meta)
            self._live.pop(run_id, None)
            logging.getLogger().removeHandler(handler)
            handler.close()
            with self._lock:
                self._active = None
                self._cancel.pop(run_id, None)

    # --- persistence ----------------------------------------------------------------------

    def _save(self, meta: dict) -> None:
        path = self.root / meta["id"] / "run.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=1, default=str))
        tmp.replace(path)

    def _recover_interrupted(self) -> None:
        for meta in self.list():
            if meta["status"] in ACTIVE:
                meta.update(status="failed", error="The console stopped while this run was in progress.",
                            finished_at=meta.get("finished_at") or _now())
                _mark_unfinished(meta)
                self._save(meta)

    def import_cli_output(self, output_dir: Path, images_dir: Path, input_file: Path) -> str | None:
        """Register the CLI's last run as a console run (once), so results are browsable immediately."""
        manifest_path = output_dir / "run_manifest.json"
        if self.list() or not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text())
        run_id = "cli-" + manifest["generated_at"].replace(":", "").replace("-", "").replace("T", "-")[:15].lower()
        run_dir = self.root / run_id
        (run_dir / "input").mkdir(parents=True, exist_ok=True)
        if input_file.exists():
            shutil.copy2(input_file, run_dir / "input" / input_file.name)
        shutil.copytree(output_dir, run_dir / "output", dirs_exist_ok=True)
        shutil.copytree(images_dir, run_dir / "images", dirs_exist_ok=True)
        steps = {name: {"label": label, "status": "skipped", "seconds": None, "detail": "", "done": None,
                        "total": None, "item": ""} for name, label in STEPS}
        for s in manifest.get("steps", []):
            if s["step"] in steps:
                steps[s["step"]].update(status=s["status"], seconds=s["seconds"], detail=s["detail"])
        steps["write_outputs"].update(status="ok", detail="output written by the command-line run")
        settings = manifest.get("settings", {})
        self._save({
            "id": run_id, "input_name": manifest["input"]["file"], "created_at": manifest["generated_at"],
            "created_by": "command line", "status": "succeeded", "started_at": manifest["generated_at"],
            "finished_at": manifest["generated_at"], "error": "", "current_step": None, "steps": steps,
            "options": {"skus": settings.get("skus") or [], "image_provider": settings.get("image_provider", "auto"),
                        "image_limit": settings.get("image_limit"), "skip_images": False, "skip_competitors": False,
                        "competitor_limit": settings.get("competitor_limit", 0), "use_cache": True},
            "imported": True,
        })
        return run_id


def _mark_unfinished(meta: dict) -> None:
    """The step that was in flight stopped there; steps after it never started."""
    for name, step in meta["steps"].items():
        if step["status"] == "running":
            step["status"] = "failed" if meta["status"] == "failed" else "cancelled"
            step["detail"] = step["detail"] or "stopped here"
            meta["stopped_at"] = name
        elif step["status"] == "pending":
            step["status"] = "not_started"


class _RunLogHandler(logging.Handler):
    """Captures log lines emitted by one run's threads (the worker and its pools)."""

    def __init__(self, buffer: deque, path: Path) -> None:
        super().__init__(logging.INFO)
        self.buffer = buffer
        self.file = path.open("a")
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        self.thread_names = {threading.current_thread().name}

    def emit(self, record: logging.LogRecord) -> None:
        name = record.threadName or ""
        if not (name in self.thread_names or name.startswith("ThreadPoolExecutor")):
            return
        line = self.format(record).split("\n")[0]
        self.buffer.append(line)
        self.file.write(line + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()
        super().close()


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "workbook"
    return stem if stem.lower().endswith(".xlsx") else f"{stem}.xlsx"


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")
