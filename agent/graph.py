"""The agent: a LangGraph state machine that runs the whole chain from one input file.

load_input -> analyse_products -> generate_content -> generate_images -> research_competitors
           -> analyse_sales -> write_outputs

Each node reads what it needs from the state and adds its results. Nodes are wrapped so that a
failure in one step is recorded in the run log and the chain continues (a broken image API must not
cost the sales analysis); only an unreadable input file stops the run.
"""
from __future__ import annotations

import logging
import operator
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from .cache import DiskCache
from .competitors import CompetitorRow, CompetitorSummary, run_competitors
from .config import Settings
from .content import ContentResult, generate_content
from .images import ImageGenerator, ImageResult
from .ingest import InputError, load_workbook
from .llm import LLM
from .models import Product
from .pricing import OurPrice
from .quality import QualityLog
from .sales_analysis import SalesReport, analyse_sales, generate_insights
from .sales_clean import clean_sales
from .vendor import clean_vendor

log = logging.getLogger(__name__)


class StepRecord(TypedDict):
    step: str
    status: str  # ok | skipped | failed
    seconds: float
    detail: str


class PipelineState(TypedDict, total=False):
    vendor_raw: pd.DataFrame
    sales_raw: pd.DataFrame
    products: list[Product]
    content: dict[str, ContentResult]
    images: dict[str, ImageResult]
    competitor_summaries: list[CompetitorSummary]
    competitor_rows: list[CompetitorRow]
    our_prices: dict[str, OurPrice]
    sales_lines: pd.DataFrame
    sales_report: SalesReport
    outputs: dict[str, str]
    fatal_error: str
    steps: Annotated[list[StepRecord], operator.add]


class RunCancelled(Exception):
    """Raised between work items once a cancel was requested."""


@dataclass
class RunContext:
    """Services shared by the nodes. Kept out of the graph state because they are not data."""

    settings: Settings
    cache: DiskCache
    quality: QualityLog = field(default_factory=QualityLog)
    _llm: LLM | None = None
    on_event: Callable[[dict], None] | None = None  # progress sink (the web console); CLI leaves it unset
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM(self.settings, self.cache)
        return self._llm

    def emit(self, **event) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:  # a broken progress sink must never break a run
                log.exception("progress sink failed")

    def progress(self, step: str, done: int, total: int, label: str = "") -> None:
        self.emit(type="progress", step=step, done=done, total=total, label=label)
        if self.cancel.is_set():
            raise RunCancelled()


def step(name: str, *, fatal: bool = False) -> Callable:
    def wrap(fn: Callable[[PipelineState, RunContext], dict]) -> Callable:
        def node(state: PipelineState, ctx: RunContext) -> dict:
            t0 = time.monotonic()
            if ctx.cancel.is_set():
                return _finish(ctx, {}, _record(name, "cancelled", t0, "run cancelled"))
            log.info("> %s", name)
            ctx.emit(type="step", step=name, status="running")
            try:
                update = fn(state, ctx) or {}
                detail = update.pop("_detail", "")
                status = update.pop("_status", "ok")
            except RunCancelled:
                return _finish(ctx, {}, _record(name, "cancelled", t0, "cancelled by user"))
            except InputError as exc:
                return _finish(ctx, {"fatal_error": str(exc)}, _record(name, "failed", t0, str(exc)))
            except Exception as exc:
                if fatal:
                    ctx.emit(type="step", step=name, status="failed", detail=f"{type(exc).__name__}: {exc}")
                    raise
                log.exception("Step %s failed; continuing", name)
                return _finish(ctx, {}, _record(name, "failed", t0, f"{type(exc).__name__}: {exc}"))
            log.info("  %s %s in %.1fs %s", name, status, time.monotonic() - t0, detail)
            return _finish(ctx, update, _record(name, status, t0, detail))

        node.__name__ = name
        return node

    return wrap


def _finish(ctx: RunContext, update: dict, record: StepRecord) -> dict:
    ctx.emit(type="step", **record)
    return {**update, "steps": [record]}


def _record(name: str, status: str, t0: float, detail: str) -> StepRecord:
    return {"step": name, "status": status, "seconds": round(time.monotonic() - t0, 1), "detail": detail}


# --- nodes ----------------------------------------------------------------------------------


@step("load_input", fatal=True)
def load_input(state: PipelineState, ctx: RunContext) -> dict:
    vendor, sales = load_workbook(ctx.settings.input_path)
    return {"vendor_raw": vendor, "sales_raw": sales,
            "_detail": f"{len(vendor)} vendor rows, {len(sales)} sales lines"}


@step("analyse_products")
def analyse_products(state: PipelineState, ctx: RunContext) -> dict:
    products = clean_vendor(state["vendor_raw"], ctx.quality)
    if ctx.settings.skus:
        wanted = set(ctx.settings.skus)
        products = [p for p in products if p.sku in wanted]
    return {"products": products, "_detail": f"{len(products)} SKUs ready"}


@step("generate_content")
def content_node(state: PipelineState, ctx: RunContext) -> dict:
    products = state.get("products", [])
    done = [0]
    lock = threading.Lock()
    llm = ctx.llm

    def one(p: Product) -> ContentResult:
        if ctx.cancel.is_set():
            raise RunCancelled()
        result = generate_content(p, llm)
        with lock:
            done[0] += 1
            n = done[0]
        ctx.progress("generate_content", n, len(products), p.sku)
        return result

    ctx.progress("generate_content", 0, len(products))
    with ThreadPoolExecutor(max_workers=ctx.settings.max_workers) as pool:
        results = list(pool.map(one, products))
    content = {r.sku: r for r in results}
    counts = pd.Series([r.qa_status for r in results]).value_counts().to_dict()
    return {"content": content, "_detail": ", ".join(f"{k}: {v}" for k, v in counts.items())}


@step("generate_images")
def images_node(state: PipelineState, ctx: RunContext) -> dict:
    products = state.get("products", [])
    if ctx.settings.skip_images:
        return {"images": {}, "_status": "skipped", "_detail": "--skip-images"}
    limit = ctx.settings.image_limit
    gen = ImageGenerator(ctx.settings)
    ctx.emit(type="progress", step="generate_images", done=0, total=len(products), label="downloading source photos")
    gen.prefetch(products if limit is None else products[:limit])
    images = {}
    for i, p in enumerate(products):
        ctx.progress("generate_images", i, len(products), p.sku)
        if limit is not None and i >= limit:
            images[p.sku] = ImageResult(sku=p.sku, status="skipped", source_url=p.image_url,
                                        note=f"Outside --image-limit {limit} (code supports the full set)")
            continue
        images[p.sku] = gen.generate(p)
        log.info("  [%d/%d] %s: %s %s", i + 1, len(products), p.sku, images[p.sku].status, images[p.sku].note)
    ok = sum(r.status == "ok" for r in images.values())
    return {"images": images, "_detail": f"{ok}/{len(products)} images written"}


@step("research_competitors")
def competitors_node(state: PipelineState, ctx: RunContext) -> dict:
    if ctx.settings.skip_competitors:
        return {"_status": "skipped", "_detail": "--skip-competitors"}
    summaries, rows, prices = run_competitors(
        state.get("products", []), state.get("content", {}), ctx.settings, ctx.cache, ctx.llm,
        on_progress=lambda done, total, label: ctx.progress("research_competitors", done, total, label))
    priced = sum(1 for p in prices.values() if p.price)
    return {"competitor_summaries": summaries, "competitor_rows": rows, "our_prices": prices,
            "_detail": f"{len(summaries)} models benchmarked, {len(rows)} competitor rows, {priced} SKUs priced on storefront"}


@step("analyse_sales")
def sales_node(state: PipelineState, ctx: RunContext) -> dict:
    lines = clean_sales(state["sales_raw"], ctx.quality)
    report = analyse_sales(lines)
    try:
        llm = ctx.llm
    except Exception as exc:  # no LLM configured: insights fall back to rules
        log.warning("No LLM for insights: %s", exc)
        llm = None
    generate_insights(report, llm)
    return {"sales_lines": lines, "sales_report": report,
            "_detail": f"net sales ${report.summary['net_sales']:,.2f}, insights: {report.insights_source}"}


@step("write_outputs", fatal=True)
def outputs_node(state: PipelineState, ctx: RunContext) -> dict:
    from .outputs import write_outputs

    paths = write_outputs(state, ctx)
    return {"outputs": paths, "_detail": ", ".join(paths.values())}


# --- graph ----------------------------------------------------------------------------------


def build_graph(ctx: RunContext):
    g = StateGraph(PipelineState)
    nodes: list[tuple[str, Callable]] = [
        ("load_input", load_input),
        ("analyse_products", analyse_products),
        ("generate_content", content_node),
        ("generate_images", images_node),
        ("research_competitors", competitors_node),
        ("analyse_sales", sales_node),
        ("write_outputs", outputs_node),
    ]
    for name, fn in nodes:
        g.add_node(name, _bind(fn, ctx))
    g.add_edge(START, "load_input")
    g.add_conditional_edges("load_input", lambda s: END if s.get("fatal_error") else "analyse_products",
                            {END: END, "analyse_products": "analyse_products"})
    for (a, _), (b, _) in zip(nodes[1:], nodes[2:]):
        g.add_edge(a, b)
    g.add_edge("write_outputs", END)
    return g.compile()


def _bind(fn: Callable, ctx: RunContext) -> Callable[[PipelineState], Any]:
    def node(state: PipelineState):
        return fn(state, ctx)

    node.__name__ = fn.__name__
    return node
