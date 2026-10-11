"""'Our price' = the live price on Benzara's own storefront (benzara.com, Shopify).

Why not the sales sheet: the 25 vendor SKUs never appear in it, and its itemName column holds codes,
not product names, so there is nothing to match a reference price on. The storefront is the
company's own published price for the exact SKU, which is the most defensible reference.
"""
from __future__ import annotations

import datetime as dt
import logging

import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .cache import DiskCache
from .config import Settings

log = logging.getLogger(__name__)
STOREFRONT = "benzara.com"


class OurPrice(BaseModel):
    sku: str
    price: float | None = None
    currency: str = "USD"
    source: str = STOREFRONT
    url: str | None = None
    listing_title: str | None = None
    in_stock: bool | None = None
    fetched_at: str
    note: str = ""


def lookup_our_price(sku: str, settings: Settings, cache: DiskCache) -> OurPrice:
    today = dt.date.today().isoformat()
    key = DiskCache.key(sku, today)  # prices change: cache per day
    if (hit := cache.get("storefront", key)) is not None:
        return OurPrice.model_validate(hit)
    try:
        products = _search(settings.storefront_search_url, sku)
    except requests.RequestException as exc:
        log.warning("Storefront lookup failed for %s: %s", sku, exc)
        return OurPrice(sku=sku, fetched_at=today, note=f"Storefront lookup failed: {type(exc).__name__}")

    match = next((p for p in products if sku.lower() in (p.get("title", "") + p.get("url", "")).lower()), None)
    if match is None:
        result = OurPrice(sku=sku, fetched_at=today, note=f"Not listed on {STOREFRONT}")
    else:
        result = OurPrice(
            sku=sku,
            price=float(match["price"]),
            url=f"https://{STOREFRONT}{match['url'].split('?')[0]}",
            listing_title=match.get("title"),
            in_stock=match.get("available"),
            fetched_at=today,
            note="" if match.get("available") else "Listed but currently out of stock",
        )
    cache.set("storefront", key, result.model_dump())
    return result


@retry(retry=retry_if_exception_type(requests.RequestException), stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _search(url: str, sku: str) -> list[dict]:
    resp = requests.get(
        url,
        params={"q": sku, "resources[type]": "product", "resources[limit]": 5},
        headers={"User-Agent": "Mozilla/5.0 (catalog-agent)"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("resources", {}).get("results", {}).get("products", [])
