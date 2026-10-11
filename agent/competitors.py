"""Competitor benchmarking.

  targets   one representative SKU per model (collection + product type); colour variants share
            the same competitive set, so searching each variant would only spend credits.
  search    Google Shopping via SerpAPI (real listings with marketplace, price and link).
  filter    drop our own listings, non-US shops, second-hand items and implausible prices.
  select    the LLM picks the most comparable listings BY ID from the retrieved candidates
            (it cannot invent a product or a price) and explains comparability.
  compute   price differences, competitor median and positioning are deterministic.
  insight   LLM writes a short insight from the computed numbers; numbers are checked.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from collections import OrderedDict

import requests
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .cache import DiskCache
from .config import Settings
from .content import ContentResult
from .grounding import ungrounded
from .llm import LLM, LLMError
from .models import Product
from .pricing import OurPrice, lookup_our_price

log = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search.json"
MAJOR_MARKETPLACES = ("amazon", "wayfair", "walmart", "home depot", "lowe's", "target", "overstock",
                      "bed bath & beyond", "macy's", "kohl's", "costco", "ashley", "ikea", "houzz", "best buy")
OWN_BRAND_RE = re.compile(r"benjara|benzara", re.I)
NON_US_SHOP_RE = re.compile(r"\.(it|pt|de|fr|es|nl|pl|ca|au|in|co\.uk|uk|ie|be|ch|at|se|dk|no|fi|mx|br)$", re.I)
MULTI_PACK_RE = re.compile(r"\b(set of [2-9]|[2-9]\s*-?\s*(pcs|pieces|pack)|[2-9]\s*-?\s*piece)\b", re.I)
PRICE_BAND = (0.25, 4.0)   # candidate price must be within this multiple of our price
MAX_CANDIDATES = 25
POSITION_BAND_PCT = 15     # +/- vs competitor median counts as "at market"
LAUNCH_BAND_PCT = 10       # unlisted SKUs: suggested launch band = competitor median +/- 10%


class Candidate(BaseModel):
    id: int
    title: str
    source: str
    price: float
    link: str | None = None
    rating: float | None = None
    reviews: int | None = None
    major_marketplace: bool = False


class Pick(BaseModel):
    candidate_id: int
    comparability: str = Field(description="Max 20 words: why this is a fair like-for-like comparison")


class CompetitorPicks(BaseModel):
    picks: list[Pick] = Field(min_length=1, max_length=5)


class CompetitorInsight(BaseModel):
    insight: str = Field(description="2 sentences: where our price sits, then one concrete pricing or listing action")


class CompetitorRow(BaseModel):
    sku: str
    covers_skus: list[str]
    competitor_product: str
    marketplace: str
    competitor_price: float
    our_price: float | None
    price_diff_usd: float | None
    price_diff_pct: float | None
    comparability: str
    link: str | None
    rating: float | None = None
    reviews: int | None = None


class CompetitorSummary(BaseModel):
    sku: str
    covers_skus: list[str]
    product: str
    search_query: str
    our_price: OurPrice
    competitor_min: float | None = None
    competitor_median: float | None = None
    competitor_max: float | None = None
    positioning: str = ""
    suggested_price_band: str = ""
    insight: str = ""
    insight_source: str = ""
    selection_source: str = ""
    candidates_found: int = 0
    candidates_after_filter: int = 0
    status: str = "ok"  # ok | no_results | error
    error: str = ""


# --- targets --------------------------------------------------------------------------------


def select_targets(products: list[Product], our_prices: dict[str, OurPrice], settings: Settings) -> list[tuple[Product, list[str]]]:
    """One representative per (collection, product type); prefer a variant with a live price."""
    groups: OrderedDict[tuple, list[Product]] = OrderedDict()
    for p in products:
        groups.setdefault((p.collection or p.sku, p.product_type), []).append(p)
    targets = []
    for members in groups.values():
        priced = [m for m in members if our_prices.get(m.sku) and our_prices[m.sku].price]
        rep = (priced or members)[0]
        targets.append((rep, [m.sku for m in members]))
    if settings.competitor_skus:
        wanted = set(settings.competitor_skus)
        targets = [t for t in targets if wanted & set(t[1])]
    # Priced SKUs first: they produce a real price comparison; unpriced ones get a suggested band.
    targets.sort(key=lambda t: our_prices.get(t[0].sku) is None or our_prices[t[0].sku].price is None)
    return targets[: settings.competitor_limit] if settings.competitor_limit else targets


def build_query(product: Product, content: ContentResult | None) -> str:
    """Shopping query from the generated title minus brand-specific words (collection, colour)."""
    if content and content.title:
        segments = [s.strip() for s in content.title.split(",")]
        colour_words = {w.lower() for c in product.colors for w in c.split()}
        if len(segments) > 1 and colour_words & {w.lower() for w in segments[-1].split()}:
            segments = segments[:-1]  # trailing "Black Leather Aire" segment
        query = " ".join(segments)
        if product.collection:
            query = re.sub(rf"\b{re.escape(product.collection)}\b", "", query, flags=re.I)
        query = re.sub(r"\b(with|and|in)\b", " ", query, flags=re.I)
        return re.sub(r"\s+", " ", query).strip()
    return " ".join(filter(None, [product.product_type, product.materials[0] if product.materials else None]))


# --- search + filter ------------------------------------------------------------------------


def search_shopping(query: str, settings: Settings, cache: DiskCache) -> list[dict]:
    key = DiskCache.key(query, "google_shopping", "us")
    if (hit := cache.get("serpapi", key)) is not None:
        return hit
    if not settings.serpapi_api_key:
        raise RuntimeError("SERPAPI_API_KEY not set")
    data = _serpapi(query, settings.serpapi_api_key)
    if data.get("error") and "hasn't returned any results" not in data["error"]:
        raise RuntimeError(f"SerpAPI: {data['error']}")
    results = data.get("shopping_results", [])
    cache.set("serpapi", key, results)
    return results


@retry(retry=retry_if_exception_type(requests.RequestException), stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
def _serpapi(query: str, api_key: str) -> dict:
    resp = requests.get(SERPAPI_URL, params={"engine": "google_shopping", "q": query, "gl": "us", "hl": "en",
                                             "api_key": api_key}, timeout=60)
    if resp.status_code >= 500:
        resp.raise_for_status()
    return resp.json()


def filter_candidates(raw: list[dict], product: Product, our_price: float | None) -> list[Candidate]:
    out: list[Candidate] = []
    seen_titles: set[str] = set()
    for r in raw:
        price = r.get("extracted_price")
        title = (r.get("title") or "").strip()
        source = (r.get("source") or "").strip()
        if not price or price <= 0 or not title:
            continue
        if OWN_BRAND_RE.search(title) or OWN_BRAND_RE.search(source) or product.sku.lower() in title.lower():
            continue  # our own product resold elsewhere is not a competitor
        if product.collection and re.search(rf"\b{re.escape(product.collection)}\b", title, re.I):
            continue
        if r.get("second_hand_condition") or NON_US_SHOP_RE.search(source):
            continue
        if MULTI_PACK_RE.search(title) and not (product.included and MULTI_PACK_RE.search(product.included)):
            continue
        if our_price and not (PRICE_BAND[0] * our_price <= price <= PRICE_BAND[1] * our_price):
            continue
        norm = re.sub(r"\W+", " ", title.lower())[:60]
        if norm in seen_titles:
            continue
        seen_titles.add(norm)
        out.append(Candidate(
            id=len(out) + 1, title=title, source=source, price=float(price),
            link=r.get("product_link") or r.get("link"), rating=r.get("rating"), reviews=r.get("reviews"),
            major_marketplace=any(m in source.lower() for m in MAJOR_MARKETPLACES),
        ))
        if len(out) >= MAX_CANDIDATES:
            break
    return out


# --- selection + insight --------------------------------------------------------------------

SELECT_SYSTEM = (
    "You are a pricing analyst for Benjara, a US furniture brand. You pick fair like-for-like competitor products "
    "from a list of real Google Shopping listings. Reference candidates only by their id."
)
INSIGHT_SYSTEM = (
    "You are a pricing analyst for Benjara, a US furniture brand. Write a 2-sentence competitive insight for the "
    "commercial team: where our price sits against the competitors, then one concrete pricing or listing action. "
    "Use only numbers present in the DATA (you may round them); do not calculate new ones. Quote the gap to the "
    "median from our_price_minus_median_usd / our_price_vs_median_pct; never describe our price itself as the premium. "
    "Write money as $1,234.56."
)


def benchmark(product: Product, covers: list[str], content: ContentResult | None, our: OurPrice,
              settings: Settings, cache: DiskCache, llm: LLM) -> tuple[CompetitorSummary, list[CompetitorRow]]:
    query = build_query(product, content)
    title = content.title if content and content.title else product.product_type or product.sku
    summary = CompetitorSummary(sku=product.sku, covers_skus=covers, product=title, search_query=query, our_price=our)
    try:
        raw = search_shopping(query, settings, cache)
    except Exception as exc:
        summary.status, summary.error = "error", f"Search failed: {exc}"
        log.error("Competitor search failed for %s: %s", product.sku, exc)
        return summary, []
    candidates = filter_candidates(raw, product, our.price)
    summary.candidates_found, summary.candidates_after_filter = len(raw), len(candidates)
    if not candidates:
        summary.status = "no_results"
        summary.insight = "No comparable listings found on Google Shopping for this query."
        return summary, []

    picks, summary.selection_source = _pick(product, our, candidates, settings.competitors_per_sku, llm)
    by_id = {c.id: c for c in candidates}
    rows = []
    for pick in picks:
        c = by_id[pick.candidate_id]
        diff = round(our.price - c.price, 2) if our.price else None
        rows.append(CompetitorRow(
            sku=product.sku, covers_skus=covers, competitor_product=c.title, marketplace=c.source,
            competitor_price=c.price, our_price=our.price, price_diff_usd=diff,
            price_diff_pct=round(diff / c.price * 100, 1) if diff is not None else None,
            comparability=pick.comparability, link=c.link, rating=c.rating, reviews=c.reviews,
        ))
    prices = [r.competitor_price for r in rows]
    summary.competitor_min, summary.competitor_max = min(prices), max(prices)
    summary.competitor_median = round(statistics.median(prices), 2)
    summary.positioning = positioning(our.price, summary.competitor_median)
    if our.price is None:
        lo, hi = launch_band(summary.competitor_median)
        summary.suggested_price_band = f"${lo:,.2f} - ${hi:,.2f}"
    summary.insight, summary.insight_source = _insight(summary, rows, llm)
    return summary, rows


def positioning(our_price: float | None, median: float | None) -> str:
    if our_price is None or not median:
        return "not listed (no our-price)"
    gap = (our_price / median - 1) * 100
    if gap > POSITION_BAND_PCT:
        return f"premium (+{gap:.0f}% vs median)"
    if gap < -POSITION_BAND_PCT:
        return f"value ({gap:.0f}% vs median)"
    return f"at market ({gap:+.0f}% vs median)"


def launch_band(median: float) -> tuple[float, float]:
    return round(median * (1 - LAUNCH_BAND_PCT / 100), 2), round(median * (1 + LAUNCH_BAND_PCT / 100), 2)


def _pick(product: Product, our: OurPrice, candidates: list[Candidate], n: int, llm: LLM) -> tuple[list[Pick], str]:
    listing = "\n".join(
        f"[{c.id}] {c.title} | {c.source}{' (major marketplace)' if c.major_marketplace else ''} | ${c.price:,.2f}"
        for c in candidates
    )
    facts = {
        "product_type": product.product_type, "included": product.included, "materials": product.materials,
        "dimensions": product.dimensions.label() if product.dimensions else None, "key_features": product.features[:6],
    }
    feedback = ""
    for _ in range(2):
        prompt = (
            f"OUR PRODUCT: {facts}\n\nCANDIDATES:\n{listing}\n\n"
            f"Pick the {n} most comparable candidates: same product type and quantity first, then similar size, "
            "features and materials. Prefer major marketplaces when equally comparable. Avoid accessories, parts "
            f"and bundles.{feedback}"
        )
        try:
            out = llm.structured(system=SELECT_SYSTEM, prompt=prompt, schema=CompetitorPicks,
                                 namespace="competitor_picks", temperature=0.1)
        except LLMError as exc:
            log.warning("Competitor selection failed for %s: %s", product.sku, exc)
            break
        ids = [p.candidate_id for p in out.value.picks]
        valid = {c.id for c in candidates}
        if all(i in valid for i in ids) and len(set(ids)) == len(ids):
            return out.value.picks[:n], f"{out.provider}:{out.model}"
        log.info("%s: invalid competitor ids %s; retrying", product.sku, ids)
        feedback = f"\n\nYour previous ids {ids} were invalid or repeated. Use distinct ids from the list."
    ranked = sorted(candidates, key=lambda c: (not c.major_marketplace, c.id))[:n]
    return [Pick(candidate_id=c.id, comparability="Top Google Shopping match for the same product type") for c in ranked], "rule-based fallback"


def _insight(summary: CompetitorSummary, rows: list[CompetitorRow], llm: LLM) -> tuple[str, str]:
    our = summary.our_price
    data = {
        "our_product": summary.product,
        "our_price_usd": our.price if our.price else "not listed on benzara.com yet",
        "our_storefront_stock": {True: "in stock", False: "out of stock"}.get(our.in_stock, "not listed"),
        "competitors": [
            {"product": r.competitor_product[:90], "marketplace": r.marketplace, "price_usd": r.competitor_price,
             "our_price_minus_competitor_usd": r.price_diff_usd, "our_price_vs_competitor_pct": r.price_diff_pct}
            for r in rows
        ],
        "competitor_median_usd": summary.competitor_median,
        "competitor_min_usd": summary.competitor_min,
        "competitor_max_usd": summary.competitor_max,
        "position_vs_median": summary.positioning,
    }
    if our.price and summary.competitor_median:
        data["our_price_minus_median_usd"] = round(our.price - summary.competitor_median, 2)
        data["our_price_vs_median_pct"] = round((our.price / summary.competitor_median - 1) * 100, 1)
    if summary.suggested_price_band:
        lo, hi = launch_band(summary.competitor_median)
        data["suggested_launch_band_usd"] = {"low": lo, "high": hi, "rule": f"competitor median +/- {LAUNCH_BAND_PCT}%"}
    feedback = ""
    for _ in range(2):
        try:
            out = llm.structured(system=INSIGHT_SYSTEM, prompt=f"DATA (JSON):\n{json.dumps(data, indent=1)}{feedback}",
                                 schema=CompetitorInsight, namespace="competitor_insight", temperature=0.3)
        except LLMError as exc:
            log.warning("Competitor insight failed for %s: %s", summary.sku, exc)
            break
        bad = ungrounded(out.value.insight, data)
        if not bad:
            return out.value.insight.strip(), f"{out.provider}:{out.model}"
        log.info("%s: competitor insight used ungrounded numbers %s; retrying", summary.sku, bad)
        feedback = f"\n\nYour previous answer used numbers not in the DATA: {bad}. Use only DATA numbers."
    return _template_insight(summary), "rule-based fallback"


def _template_insight(s: CompetitorSummary) -> str:
    if s.our_price.price:
        return (f"At ${s.our_price.price:,.2f} we are {s.positioning} of ${s.competitor_median:,.2f} across the selected "
                f"competitors (${s.competitor_min:,.2f}-${s.competitor_max:,.2f}). Review whether our feature set justifies the gap.")
    return (f"Comparable listings sell for ${s.competitor_min:,.2f}-${s.competitor_max:,.2f} (median ${s.competitor_median:,.2f}). "
            f"Suggested launch band: {s.suggested_price_band}.")


def run_competitors(products: list[Product], contents: dict[str, ContentResult], settings: Settings,
                    cache: DiskCache, llm: LLM, on_progress=None) -> tuple[list[CompetitorSummary], list[CompetitorRow], dict[str, OurPrice]]:
    report = on_progress or (lambda done, total, label: None)
    our_prices = {}
    for i, p in enumerate(products):
        report(i, len(products), f"storefront price {p.sku}")
        our_prices[p.sku] = lookup_our_price(p.sku, settings, cache)
    summaries, rows = [], []
    targets = select_targets(products, our_prices, settings)
    for i, (product, covers) in enumerate(targets, 1):
        report(i - 1, len(targets), f"benchmarking {product.sku}")
        s, r = benchmark(product, covers, contents.get(product.sku), our_prices[product.sku], settings, cache, llm)
        log.info("  [%d/%d] %s: %s, %d competitors, %s", i, len(targets), product.sku, s.status, len(r), s.positioning)
        summaries.append(s)
        rows += r
    return summaries, rows, our_prices
