"""Catalog Agent console: FastAPI app with server-rendered pages."""
from __future__ import annotations

import datetime as dt
import io
import logging
import os
import re
import tempfile
import warnings
import zipfile
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from PIL import Image
from starlette.middleware.sessions import SessionMiddleware

from agent.config import PROJECT_ROOT
from agent.ingest import InputError, load_workbook

from . import auth
from .providers import test_provider
from .results import load_results
from .runs import ACTIVE, RUNS_DIR, STEPS, RunBusy, RunManager, RunOptions
from .store import IMAGE_PROVIDERS, SECRETS, SettingsStore

load_dotenv(PROJECT_ROOT / ".env")
# Pipeline modules log at INFO; without this the root logger drops them and run logs stay empty.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
for _noisy in ("httpx", "httpcore", "urllib3", "google_genai", "groq", "uvicorn.access"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

HERE = Path(__file__).parent
SAMPLE_WORKBOOK = PROJECT_ROOT / "data" / "test_data.xlsx"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SKU_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$")
RESULT_TABS = ["overview", "content", "images", "competitors", "sales", "quality", "files", "log"]

store = SettingsStore()
runs = RunManager(store, root=Path(os.getenv("CONSOLE_RUNS_DIR") or RUNS_DIR))
accounts = auth.Accounts()
if os.getenv("CONSOLE_IMPORT_CLI", "1") != "0":
    runs.import_cli_output(PROJECT_ROOT / "output", PROJECT_ROOT / "images", SAMPLE_WORKBOOK)

app = FastAPI(title="Catalog Agent", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(SessionMiddleware, secret_key=auth.session_secret(), session_cookie="catalog_agent",
                   max_age=12 * 3600, same_site="lax", https_only=False)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


# --- template helpers -----------------------------------------------------------------------


def _money(v) -> str:
    return "" if v is None else f"${v:,.2f}"


def _num(v, digits: int = 0) -> str:
    if v is None:
        return ""
    return f"{v:,.{digits}f}"


def _when(iso: str | None) -> str:
    if not iso:
        return ""
    t = dt.datetime.fromisoformat(iso)
    today = dt.date.today()
    if t.date() == today:
        return f"Today {t:%H:%M}"
    if t.date() == today - dt.timedelta(days=1):
        return f"Yesterday {t:%H:%M}"
    return f"{t:%d %b %Y, %H:%M}"


def _duration(run: dict) -> str:
    if not run.get("started_at") or run.get("imported"):
        return ""
    end = dt.datetime.fromisoformat(run["finished_at"]) if run.get("finished_at") else dt.datetime.now()
    secs = max(0, int((end - dt.datetime.fromisoformat(run["started_at"])).total_seconds()))
    return f"{secs // 60}m {secs % 60:02d}s" if secs >= 60 else f"{secs}s"


def _model_name(raw: str | None) -> str:
    """'gemini:gemini-2.5-flash (cached)' -> 'Gemini 2.5 Flash · cached'; raw IDs stay available as a tooltip."""
    if not raw:
        return "—"
    cached = "(cached)" in raw
    base = raw.replace("(cached)", "").strip()
    provider, _, model = base.partition(":")
    if provider == "composite":
        name = "Local composite"
    elif provider == "gemini":
        words = model.removeprefix("gemini-").split("-")
        name = "Gemini " + " ".join(w if w[:1].isdigit() else w.capitalize() for w in words)
    elif provider == "groq":
        short = model.split("/")[-1]
        name = ("GPT-OSS " + short.split("-")[-1].upper() if short.startswith("gpt-oss") else short) + " via Groq"
    else:
        name = base[:1].upper() + base[1:]
    return f"{name} · cached" if cached else name


def _prov(raw: str | None) -> Markup:
    return Markup('<span title="{}">{}</span>').format(raw or "", _model_name(raw))


_MODEL_TOKEN = re.compile(r"\b(?:gemini|groq|composite):[\w./+-]+(?: \(cached\))?")


def _prov_text(text: str | None) -> str:
    """Rewrite provider:model tokens inside free text (step results) into readable names."""
    return _MODEL_TOKEN.sub(lambda m: _model_name(m.group(0)), text or "")


def _secs(v) -> str:
    """One duration format everywhere: 0.4s, 12.3s, 102s; an em dash when no time was recorded."""
    if v is None:
        return "—"
    return f"{v:.1f}s" if v < 10 else f"{v:.0f}s"


def _bytes(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n > 1024 * 1024 else f"{n / 1024:.0f} KB"


def _weekly_chart(weekly: list[dict]) -> dict | None:
    """Column chart data for the weekly net-sales chart, in percentages of the plot height.

    Rendered as HTML columns rather than a scaled SVG so axis text stays at its real size at every width.
    """
    if not weekly:
        return None
    top_value = max(w["net_sales"] for w in weekly)
    step = _nice_step(top_value / 4)
    y_max = step * (int(top_value // step) + 1)
    peak = max(range(len(weekly)), key=lambda i: weekly[i]["net_sales"])
    bars = []
    for i, w in enumerate(weekly):
        start = dt.date.fromisoformat(w["week_start"])
        bars.append({
            "pct": round(w["net_sales"] / y_max * 100, 2), "label": f"{start:%d %b}", "value": w["net_sales"],
            "orders": w["orders"], "units": w["units"], "partial": w["partial"], "days": w["days_covered"],
            "peak": i == peak, "range": f"{start:%d %b} – {dt.date.fromisoformat(w['week_end']):%d %b}",
        })
    ticks = [{"pct": round(v / y_max * 100, 2), "label": "$0" if v == 0 else f"${v / 1000:,.0f}k"}
             for v in [step * k for k in range(int(y_max / step) + 1)]]
    return {"bars": bars, "ticks": ticks}


def _nice_step(raw: float) -> float:
    magnitude = 10 ** len(str(int(raw))) / 10
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * magnitude:
            return m * magnitude
    return 10 * magnitude


templates.env.filters.update(money=_money, num=_num, when=_when, duration=_duration, filesize=_bytes, secs=_secs,
                             prov=_prov, model_name=_model_name, prov_text=_prov_text)
templates.env.globals.update(weekly_chart=_weekly_chart)


def _asset_version() -> str:
    """Changes whenever app.css/app.js change, so browsers never run stale assets after an update."""
    return str(max(int((HERE / "static" / f).stat().st_mtime) for f in ("app.css", "app.js")))


templates.env.globals["asset_version"] = _asset_version
templates.env.globals.update(STEPS=STEPS, ACTIVE=ACTIVE, IMAGE_PROVIDERS=IMAGE_PROVIDERS)


def render(request: Request, name: str, status_code: int = 200, **ctx) -> HTMLResponse:
    user = auth.current_user(request)
    return templates.TemplateResponse(request, name, {
        "user": user, "csrf": auth.csrf_token(request), "active_run": runs.active_run,
        "nav": ctx.pop("nav", ""), **ctx,
    }, status_code=status_code)


# --- errors ---------------------------------------------------------------------------------


@app.exception_handler(auth.LoginRequired)
async def _login_required(request: Request, _exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "Your session has ended. Sign in again."}, status_code=401)
    target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    return RedirectResponse(f"/login?next={quote(target)}", status_code=303)


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
    titles = {403: "You don't have access to this", 404: "Not found"}
    return render(request, "error.html", status_code=exc.status_code,
                  title=titles.get(exc.status_code, "Something went wrong"), message=exc.detail)


def _get_run(run_id: str) -> dict:
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(404, "This run does not exist or was deleted.")
    return run


# --- auth -----------------------------------------------------------------------------------


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/"):
    if auth.current_user(request):
        return RedirectResponse(_safe_next(next), status_code=303)
    return render(request, "login.html", next=_safe_next(next), error="")


@app.post("/login")
async def login(request: Request, username: str = Form(""), password: str = Form(""), next: str = Form("/"),
                _=Depends(auth.verify_csrf)):
    client = request.client.host if request.client else "unknown"
    user, error = accounts.authenticate(username.strip(), password, client)
    if user is None:
        return render(request, "login.html", status_code=400, next=_safe_next(next), error=error, username=username)
    request.session.clear()
    request.session["user"] = {"username": user.username, "role": user.role}
    return RedirectResponse(_safe_next(next), status_code=303)


@app.post("/logout")
async def logout(request: Request, _=Depends(auth.verify_csrf)):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


def _safe_next(target: str) -> str:
    return target if target.startswith("/") and not target.startswith("//") else "/"


# --- overview and runs ----------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def overview(request: Request, user: auth.User = Depends(auth.require_user)):
    all_runs = runs.list()
    latest = all_runs[0] if all_runs else None
    # Metrics come from the newest run that wrote outputs, so a cancelled or failed latest run
    # does not blank the overview.
    metrics_run, results = None, None
    for r in all_runs[:20]:
        candidate = load_results(runs.dir(r["id"]))
        if candidate.has_output:
            metrics_run, results = r, candidate
            break
    return render(request, "overview.html", nav="overview", latest=latest, results=results, metrics_run=metrics_run,
                  recent=all_runs[:8], missing=store.missing_required())


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request, user: auth.User = Depends(auth.require_user)):
    return render(request, "runs.html", nav="runs", runs=runs.list())


@app.get("/runs/new", response_class=HTMLResponse)
def new_run_page(request: Request, user: auth.User = Depends(auth.require_user)):
    return _new_run_form(request)


def _new_run_form(request: Request, error: str = "", form: dict | None = None, status_code: int = 200):
    pipeline = store.pipeline()
    return render(request, "new_run.html", status_code=status_code, nav="new", error=error, form=form or {},
                  pipeline=pipeline, secrets=store.secret_views(), sample=SAMPLE_WORKBOOK.exists(),
                  missing=store.missing_required())


@app.post("/runs")
async def create_run(
    request: Request,
    user: auth.User = Depends(auth.require_user),
    _=Depends(auth.verify_csrf),
    source: str = Form("upload"),
    workbook: UploadFile | None = File(None),
    scope: str = Form("all"),
    skus: str = Form(""),
    image_provider: str = Form("auto"),
    images: str = Form("all"),
    image_limit: str = Form(""),
    competitors: str = Form(""),
    competitor_limit: str = Form("0"),
    use_cache: str = Form(""),
):
    form = {"source": source, "scope": scope, "skus": skus, "image_provider": image_provider, "images": images,
            "image_limit": image_limit, "competitors": competitors, "competitor_limit": competitor_limit,
            "use_cache": use_cache}

    def fail(message: str):
        return _new_run_form(request, error=message, form=form, status_code=400)

    if runs.active_run:
        return fail("Another run is in progress. Wait for it to finish or cancel it first.")
    if image_provider not in IMAGE_PROVIDERS:
        return fail("Choose an image provider from the list.")
    sku_list: list[str] = []
    if scope == "selected":
        sku_list = [s for s in re.split(r"[\s,;]+", skus.strip()) if s]
        bad = [s for s in sku_list if not SKU_RE.match(s)]
        if not sku_list or bad:
            return fail(f"Enter SKUs separated by commas or new lines{': not valid: ' + ', '.join(bad[:5]) if bad else ''}.")
    try:
        limit = int(image_limit) if images == "limit" else None
        comp_limit = int(competitor_limit or 0)
        if (limit is not None and limit < 1) or comp_limit < 0:
            raise ValueError
    except ValueError:
        return fail("Image and competitor limits must be whole numbers (image limit at least 1).")

    if source == "sample":
        if not SAMPLE_WORKBOOK.exists():
            return fail("The sample workbook is not available on this server. Upload a file instead.")
        path, name, move = SAMPLE_WORKBOOK, SAMPLE_WORKBOOK.name, False
    else:
        if workbook is None or not workbook.filename:
            return fail("Choose an .xlsx workbook to upload, or use the sample workbook.")
        if not workbook.filename.lower().endswith(".xlsx"):
            return fail(f"“{workbook.filename}” is not an .xlsx file. Export the workbook as Excel (.xlsx).")
        tmp = Path(tempfile.mkstemp(suffix=".xlsx")[1])
        size = 0
        with tmp.open("wb") as fh:
            while chunk := await workbook.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    tmp.unlink(missing_ok=True)
                    return fail("The workbook is larger than 25 MB.")
                fh.write(chunk)
        if size == 0 or tmp.read_bytes()[:2] != b"PK":
            tmp.unlink(missing_ok=True)
            return fail("The file is empty or not a valid Excel workbook.")
        path, name, move = tmp, workbook.filename, True

    try:
        load_workbook(path)  # fail fast on missing sheets / columns, before a run is created
    except InputError as exc:
        if move:
            path.unlink(missing_ok=True)
        return fail(str(exc))

    options = RunOptions(skus=sku_list, image_provider=image_provider, image_limit=limit,
                         skip_images=images == "skip", skip_competitors=competitors != "on",
                         competitor_limit=comp_limit, use_cache=use_cache == "on")
    try:
        run_id = runs.start(source=path, input_name=name, options=options, user=user.username, move=move)
    except RunBusy:
        return fail("Another run started a moment ago. Wait for it to finish.")
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str, tab: str = "overview", q: str = "", qa: str = "", notice: str = "",
               user: auth.User = Depends(auth.require_user)):
    run = _get_run(run_id)
    tab = tab if tab in RESULT_TABS else "overview"
    results = load_results(runs.dir(run_id))
    content = results.content
    if tab == "content":
        if q:
            needle = q.lower()
            content = [c for c in content if needle in (c["sku"] + " " + c["title"]).lower()]
        if qa:
            content = [c for c in content if c["qa_status"] == qa]
    return render(request, "run.html", nav="runs", run=run, tab=tab, results=results, content=content, q=q, qa=qa,
                  log=runs.log_tail(run_id, 300) if tab == "log" else [], notice=notice)


@app.get("/runs/{run_id}/products/{sku}", response_class=HTMLResponse)
def product_detail(request: Request, run_id: str, sku: str, user: auth.User = Depends(auth.require_user)):
    run = _get_run(run_id)
    results = load_results(runs.dir(run_id))
    item = next((c for c in results.content if c["sku"] == sku), None)
    if item is None:
        raise HTTPException(404, f"No content for SKU {sku} in this run.")
    skus = [c["sku"] for c in results.content]
    i = skus.index(sku)
    summary, rows = results.benchmark_for(sku)
    return render(request, "product.html", nav="runs", run=run, results=results, item=item,
                  product=results.products.get(sku, {}), image=results.images.get(sku),
                  price=results.our_price(sku), benchmark=summary, rows=rows,
                  prev_sku=skus[i - 1] if i > 0 else None, next_sku=skus[i + 1] if i + 1 < len(skus) else None)


@app.get("/api/runs/{run_id}/status")
def run_status(run_id: str, user: auth.User = Depends(auth.require_user)):
    run = _get_run(run_id)
    return {"run": run, "elapsed": _duration(run), "log": runs.log_tail(run_id, 80)}


@app.post("/runs/{run_id}/cancel")
async def cancel_run(request: Request, run_id: str, user: auth.User = Depends(auth.require_user),
                     _=Depends(auth.verify_csrf)):
    _get_run(run_id)
    runs.cancel(run_id)
    if request.headers.get("x-csrf-token"):
        return {"ok": True}
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@app.post("/runs/{run_id}/rerun")
async def rerun(run_id: str, user: auth.User = Depends(auth.require_user), _=Depends(auth.verify_csrf)):
    """Start a new run with the same workbook and settings; cached work makes it resume quickly."""
    run = _get_run(run_id)
    source = runs.dir(run_id) / "input" / run["input_name"]
    if not source.exists():
        raise HTTPException(404, "The input workbook for this run is no longer stored, so it cannot be run again.")
    fields = RunOptions.__dataclass_fields__
    options = RunOptions(**{k: v for k, v in run["options"].items() if k in fields})
    try:
        new_id = runs.start(source=source, input_name=run["input_name"], options=options, user=user.username, move=False)
    except RunBusy as busy:
        return RedirectResponse(f"/runs/{run_id}?notice=busy&active={busy.args[0]}", status_code=303)
    return RedirectResponse(f"/runs/{new_id}", status_code=303)


@app.post("/runs/{run_id}/delete")
async def delete_run(run_id: str, user: auth.User = Depends(auth.require_admin), _=Depends(auth.verify_csrf)):
    _get_run(run_id)
    try:
        runs.delete(run_id)
    except RunBusy:
        raise HTTPException(409, "Cancel this run before deleting it.")
    return RedirectResponse("/runs", status_code=303)


# --- files and media ------------------------------------------------------------------------


@app.get("/runs/{run_id}/files/{name}")
def download_file(run_id: str, name: str, user: auth.User = Depends(auth.require_user)):
    from .results import DOWNLOADS

    _get_run(run_id)
    if name not in DOWNLOADS:
        raise HTTPException(404, "Unknown file.")
    path = runs.dir(run_id) / "output" / name
    if not path.exists():
        raise HTTPException(404, "This run did not produce that file.")
    inline = name.endswith(".html")
    return FileResponse(path, filename=None if inline else name, media_type="text/html" if inline else None)


@app.get("/runs/{run_id}/input")
def download_input(run_id: str, user: auth.User = Depends(auth.require_user)):
    run = _get_run(run_id)
    path = runs.dir(run_id) / "input" / run["input_name"]
    if not path.exists():
        raise HTTPException(404, "The input workbook is not stored for this run.")
    return FileResponse(path, filename=run["input_name"])


@app.get("/runs/{run_id}/images.zip")
def download_images(run_id: str, user: auth.User = Depends(auth.require_user)):
    _get_run(run_id)
    folder = runs.dir(run_id) / "images"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for img in sorted(folder.glob("*.jpg")):
            zf.write(img, f"images/{img.name}")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="lifestyle-images-{run_id}.zip"'})


@app.get("/runs/{run_id}/images/{sku}.jpg")
def lifestyle_image(run_id: str, sku: str, w: int = 0, user: auth.User = Depends(auth.require_user)):
    _get_run(run_id)
    if not SKU_RE.match(sku):
        raise HTTPException(404, "Unknown image.")
    return _image_response(runs.dir(run_id) / "images" / f"{sku}.jpg", runs.dir(run_id) / "thumbs", f"life-{sku}", w)


@app.get("/runs/{run_id}/source/{sku}.jpg")
def source_image(run_id: str, sku: str, w: int = 0, user: auth.User = Depends(auth.require_user)):
    _get_run(run_id)
    results = load_results(runs.dir(run_id))
    url = (results.products.get(sku) or {}).get("image_url") or (results.images.get(sku) or {}).get("source_url")
    if not url:
        raise HTTPException(404, "No source photo for this SKU.")
    cached = PROJECT_ROOT / ".cache" / "source_images" / url.rsplit("/", 1)[-1]
    if not cached.exists():
        return RedirectResponse(url, status_code=302)
    return _image_response(cached, runs.dir(run_id) / "thumbs", f"src-{sku}", w)


def _image_response(path: Path, thumbs: Path, key: str, width: int):
    if not path.exists():
        raise HTTPException(404, "Image not found.")
    if width <= 0:
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
    width = min(max(width, 64), 1600)
    thumb = thumbs / f"{key}-{width}.jpg"
    if not thumb.exists() or thumb.stat().st_mtime < path.stat().st_mtime:
        thumbs.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((width, width * 4))
            im.save(thumb, "JPEG", quality=84, optimize=True)
    return FileResponse(thumb, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/sample-workbook")
def sample_workbook(user: auth.User = Depends(auth.require_user)):
    if not SAMPLE_WORKBOOK.exists():
        raise HTTPException(404, "No sample workbook on this server.")
    return FileResponse(SAMPLE_WORKBOOK, filename=SAMPLE_WORKBOOK.name)


# --- settings (admin) -----------------------------------------------------------------------


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, saved: str = "", user: auth.User = Depends(auth.require_admin)):
    return render(request, "settings.html", nav="settings", secrets=store.secret_views(), pipeline=store.pipeline(),
                  accounts=accounts.usernames(), saved=saved, error="")


@app.post("/settings/keys")
async def save_keys(request: Request, user: auth.User = Depends(auth.require_admin), _=Depends(auth.verify_csrf)):
    form = await request.form()
    changed = 0
    for key in SECRETS:
        value = str(form.get(key, "")).strip()
        if form.get(f"clear_{key}") == "on":
            store.set_secret(key, None)
            changed += 1
        elif value:
            if len(value) < 16 or any(c.isspace() for c in value):
                return render(request, "settings.html", status_code=400, nav="settings", secrets=store.secret_views(),
                              pipeline=store.pipeline(), accounts=accounts.usernames(), saved="",
                              error=f"The {SECRETS[key][0]} key looks incomplete. Paste the full key without spaces.")
            store.set_secret(key, value)
            changed += 1
    return RedirectResponse(f"/settings?saved={'keys' if changed else 'nochange'}", status_code=303)


@app.post("/settings/pipeline")
async def save_pipeline(request: Request, user: auth.User = Depends(auth.require_admin), _=Depends(auth.verify_csrf)):
    form = await request.form()
    models = [m.strip() for m in re.split(r"[\n,]+", str(form.get("text_models", ""))) if m.strip()]
    error = ""
    try:
        values = {
            "text_models": models,
            "fallback_text_model": str(form.get("fallback_text_model", "")).strip(),
            "image_model": str(form.get("image_model", "")).strip(),
            "image_provider": str(form.get("image_provider", "auto")),
            "competitor_limit": int(form.get("competitor_limit", 0) or 0),
            "competitors_per_sku": int(form.get("competitors_per_sku", 3) or 3),
            "max_workers": int(form.get("max_workers", 4) or 4),
        }
        if not models:
            error = "List at least one Gemini model."
        elif values["image_provider"] not in IMAGE_PROVIDERS:
            error = "Choose an image provider from the list."
        elif not (1 <= values["competitors_per_sku"] <= 5 and 1 <= values["max_workers"] <= 8 and values["competitor_limit"] >= 0):
            error = "Competitors per model must be 1 to 5 and parallel calls 1 to 8."
    except ValueError:
        error = "Limits must be whole numbers."
    if error:
        return render(request, "settings.html", status_code=400, nav="settings", secrets=store.secret_views(),
                      pipeline={**store.pipeline(), **{k: v for k, v in form.items()}}, accounts=accounts.usernames(),
                      saved="", error=error)
    store.set_pipeline(values)
    return RedirectResponse("/settings?saved=pipeline", status_code=303)


@app.post("/api/settings/test/{provider}")
async def test_key(provider: str, request: Request, user: auth.User = Depends(auth.require_admin),
                   _=Depends(auth.verify_csrf)):
    if provider not in SECRETS:
        raise HTTPException(404, "Unknown provider.")
    pipeline = store.pipeline()
    models = {"google_api_key": pipeline["text_models"], "groq_api_key": [pipeline["fallback_text_model"]]}.get(provider, [])
    from starlette.concurrency import run_in_threadpool

    result = await run_in_threadpool(test_provider, provider, store.secret(provider)[0], models)
    return {"ok": result.ok, "message": result.message, "detail": result.detail}
