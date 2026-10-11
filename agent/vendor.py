"""Vendor sheet -> clean `Product` records ("analyse product" step).

Vendor bullets mix several kinds of information. Each bullet is routed to where it belongs:
  - first bullet ("One tall dresser chest")      -> product type / what's included
  - "Check out our X collection"                 -> collection name (never shown as a feature)
  - "Modern, contemporary style"                 -> style
  - "Assembly required" / "Minimal assembly ..." -> assembly note
  - "41L x 37W x 41H, Seat Height: 20, in inches"-> dimensions
  - "Supports 300lbs of distributed weight"      -> weight capacity
  - everything else                              -> features
Parsing is deterministic so the facts handed to the LLM are reproducible and testable.
"""
from __future__ import annotations

import re

import pandas as pd

from .config import VENDOR_SHEET
from .models import Carton, Dimensions, Product
from .quality import QualityLog

_NUM = r"(\d+(?:\.\d+)?)"
DIMENSION_RE = re.compile(rf"{_NUM}\s*L\s*x\s*{_NUM}\s*W\s*x\s*{_NUM}\s*H", re.I)
LABELLED_DIM_RE = re.compile(rf"([A-Za-z][A-Za-z ]+?):\s*{_NUM}(?=\s*(?:,|$|in\b))", re.I)
CROSS_SELL_RE = re.compile(r"check out our\s+(.+?)\s+collection", re.I)
# "Modern, contemporary style": at most three descriptor words, nothing else in the bullet.
STYLE_RE = re.compile(r"^(?:[A-Za-z-]+(?:,\s*|\s*&\s*|\s*/\s*|\s+and\s+|\s+)){1,3}style\.?$", re.I)
ASSEMBLY_RE = re.compile(r"\bassembl", re.I)
NO_ASSEMBLY_RE = re.compile(r"\bno assembly\b", re.I)
CAPACITY_RE = re.compile(r"\bsupports?\b.*?\d+\s*(?:lbs?|pounds)\b", re.I)
COUNT_PREFIX_RE = re.compile(
    r"^(?:one|two|three|four|five|six|\d+)(?:\s*-?\s*piece)?\s+", re.I
)
TYPE_CUT_RE = re.compile(r"\s+(?:with|includes?|including|featuring)\b|,", re.I)
BULLET_COL_RE = re.compile(r"^bullet\s*(\d+)$", re.I)

_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}


def clean_vendor(df: pd.DataFrame, log: QualityLog) -> list[Product]:
    bullet_cols = sorted(
        (c for c in df.columns if BULLET_COL_RE.match(c)),
        key=lambda c: int(BULLET_COL_RE.match(c).group(1)),
    )
    _log_sheet_level(df, bullet_cols, log)

    products: list[Product] = []
    seen: set[str] = set()
    for idx, row in df.iterrows():
        sku = _text(row.get("SKU"))
        if not sku:
            log.add(VENDOR_SHEET, "SKU", "Row without SKU", "Row skipped", example=f"row {idx + 2}", severity="warning")
            continue
        if sku in seen:
            log.add(VENDOR_SHEET, "SKU", "Duplicate SKU", "Later row skipped", example=sku, severity="warning")
            continue
        seen.add(sku)
        products.append(_build_product(sku, row, bullet_cols, log))
    return products


def _build_product(sku: str, row: pd.Series, bullet_cols: list[str], log: QualityLog) -> Product:
    masked = _text(row.get("Masked SKU")) or sku
    if masked != sku:
        log.add(VENDOR_SHEET, "Masked SKU", "Masked SKU differs from SKU", "SKU used as key", example=sku)

    bullets = _dedupe([_text(row.get(c)) for c in bullet_cols], sku, log)
    parsed = _route_bullets(bullets)

    colors = _split_list(row.get("Color"), "Color", sku, log)
    materials = _split_list(row.get("Material Used"), "Material Used", sku, log)

    assembly_required = _parse_bool(row.get("Assembly Required"))
    if assembly_required is None:
        log.add(VENDOR_SHEET, "Assembly Required", "Missing or unrecognised value",
                "Left blank; assembly note from bullets used if present", example=sku, severity="warning")
    _check_assembly_conflict(sku, assembly_required, parsed["assembly"], log)

    dims = _dimensions_from_columns(row)
    bullet_dims = parsed["dims"]
    if dims is None and bullet_dims:
        dims = bullet_dims[0]
        log.add(VENDOR_SHEET, "Length1/Width1/Height1", "Product dimension columns empty",
                "Used dimensions parsed from bullet text", example=sku)
    elif dims and bullet_dims and not _same_dims(dims, bullet_dims[0]):
        log.add(VENDOR_SHEET, "Length1/Width1/Height1", "Dimension columns disagree with bullet text",
                "Kept column values; flagged for review", example=f"{sku}: {dims.label()} vs {bullet_dims[0].label()}",
                severity="warning")
    elif dims is None:
        log.add(VENDOR_SHEET, "Dimensions", "No product dimensions anywhere", "Dimensions omitted from listing",
                example=sku, severity="warning")

    weight = _positive(row.get("Weight1"))
    image_url = _text(row.get("Image URL 1"))
    if not image_url or not image_url.lower().startswith("http"):
        log.add(VENDOR_SHEET, "Image URL 1", "Missing or invalid image URL", "Lifestyle image skipped for SKU",
                example=sku, severity="warning")
        image_url = None

    product_type = parsed["product_type"]
    if not product_type:
        log.add(VENDOR_SHEET, "Bullet 1", "Could not derive product type from first bullet",
                "Fell back to Category", example=sku, severity="warning")
        product_type = _text(row.get("Category"))

    return Product(
        sku=sku,
        masked_sku=masked,
        vendor=_text(row.get("Vendor1 Name")),
        brand=_text(row.get("Brand")),
        category=_text(row.get("Category")),
        product_type=product_type,
        collection=parsed["collection"],
        style=parsed["style"],
        colors=colors,
        materials=materials,
        assembly_required=assembly_required,
        assembly_note=parsed["assembly"],
        weight_capacity=parsed["capacity"],
        product_weight_lb=weight,
        dimensions=dims,
        dimension_details=parsed["dim_details"],
        included=parsed["included"],
        features=parsed["features"],
        image_url=image_url,
        cartons=_cartons(row),
    )


def _route_bullets(bullets: list[str]) -> dict:
    out = {
        "product_type": None, "included": None, "collection": None, "style": None, "assembly": None,
        "capacity": None, "dims": [], "dim_details": [], "features": [],
    }
    for i, text in enumerate(bullets):
        if m := CROSS_SELL_RE.search(text):
            out["collection"] = m.group(1).strip().title()
        elif DIMENSION_RE.search(text):
            out["dims"].append(_dims_from_text(text))
            out["dim_details"].append(_normalise_dim_text(text))
        elif ASSEMBLY_RE.search(text):
            out["assembly"] = text
        elif STYLE_RE.match(text):
            out["style"] = re.sub(r"\s*style\.?$", "", text, flags=re.I).strip()
        elif CAPACITY_RE.search(text):
            out["capacity"] = text
        elif i == 0 and COUNT_PREFIX_RE.match(text):
            out["included"] = text
            out["product_type"] = _product_type(text)
        else:
            out["features"].append(text)
    return out


def _product_type(first_bullet: str) -> str | None:
    body = COUNT_PREFIX_RE.sub("", first_bullet, count=1)
    body = TYPE_CUT_RE.split(body, maxsplit=1)[0].strip()
    return body or None


def _dims_from_text(text: str) -> Dimensions:
    l, w, h = (float(x) for x in DIMENSION_RE.search(text).groups())
    return Dimensions(length=l, width=w, height=h)


def _normalise_dim_text(text: str) -> str:
    """'Coffee Table Dimension: 39L x 39W x 18.5H, Seat Height: 20, in inches' -> readable spec line."""
    prefix = text[: DIMENSION_RE.search(text).start()].strip(" :,-")
    prefix = re.sub(r"\s*dimensions?$", "", prefix, flags=re.I).strip()
    parts = [(f"{prefix}: " if prefix else "Overall: ") + _dims_from_text(text).label()]
    rest = text[DIMENSION_RE.search(text).end():]
    parts += [f"{label.strip()}: {float(val):g}\"" for label, val in LABELLED_DIM_RE.findall(rest)]
    return "; ".join(parts)


def _dimensions_from_columns(row: pd.Series) -> Dimensions | None:
    vals = [_positive(row.get(c)) for c in ("Length1", "Width1", "Height1")]
    if all(vals):
        return Dimensions(length=vals[0], width=vals[1], height=vals[2])
    return None


def _cartons(row: pd.Series) -> list[Carton]:
    cartons = []
    for i in range(1, 50):
        cols = [f"Ship Weight{i}", f"Ship Length{i}", f"Ship Width{i}", f"Ship Height{i}"]
        if cols[0] not in row.index:
            break
        vals = [_positive(row.get(c)) for c in cols]
        if all(vals):
            cartons.append(Carton(weight_lb=vals[0], length=vals[1], width=vals[2], height=vals[3]))
    return cartons


def _check_assembly_conflict(sku: str, required: bool | None, note: str | None, log: QualityLog) -> None:
    if required is None or not note:
        return
    note_says_none = bool(NO_ASSEMBLY_RE.search(note))
    if required == note_says_none:
        log.add(VENDOR_SHEET, "Assembly Required", "Column contradicts assembly bullet",
                "Column kept as the yes/no value; bullet surfaced as a setup note so the listing is not misleading",
                example=f"{sku}: column={'Yes' if required else 'No'}, bullet='{note}'", severity="warning")


def _log_sheet_level(df: pd.DataFrame, bullet_cols: list[str], log: QualityLog) -> None:
    filled = df[bullet_cols].apply(lambda r: sum(bool(_text(v)) for v in r), axis=1)
    log.add(VENDOR_SHEET, "Bullet 1-N", f"Ragged bullets: {filled.min()}-{filled.max()} of {len(bullet_cols)} filled per SKU",
            "Blanks dropped; bullets routed to features / dimensions / assembly / style / collection",
            count=int((filled < len(bullet_cols)).sum()))
    empty_cols = [c for c in bullet_cols if not df[c].map(_text).any()]
    if empty_cols:
        log.add(VENDOR_SHEET, ", ".join(empty_cols), "Bullet columns empty for every SKU", "Ignored",
                count=len(empty_cols))
    gaps = df[bullet_cols].apply(_has_interior_gap, axis=1).sum()
    log.add(VENDOR_SHEET, "Bullet 1-N", "Blank bullet between filled bullets", "Blank skipped, order kept", count=int(gaps))
    cats = df["Category"].map(_text).unique()
    if len(cats) == 1:
        log.add(VENDOR_SHEET, "Category", f"Category is '{cats[0]}' for every SKU (no product type)",
                "Product type derived from the first bullet ('One tall dresser chest' -> 'tall dresser chest')",
                count=len(df))


def _has_interior_gap(values: pd.Series) -> bool:
    flags = [bool(_text(v)) for v in values]
    if True not in flags:
        return False
    last = len(flags) - 1 - flags[::-1].index(True)
    return not all(flags[: last + 1])


def _dedupe(values: list[str], sku: str, log: QualityLog) -> list[str]:
    out, seen = [], set()
    for v in values:
        if not v:
            continue
        key = re.sub(r"\W+", " ", v.lower()).strip()
        if key in seen:
            log.add(VENDOR_SHEET, "Bullet 1-N", "Duplicate bullet within SKU", "Dropped duplicate", example=sku)
            continue
        seen.add(key)
        out.append(v)
    return out


def _split_list(value, field: str, sku: str, log: QualityLog) -> list[str]:
    raw = _text(value)
    parts = [p.strip() for p in raw.split(",")]
    if any(not p for p in parts) and raw:
        log.add(VENDOR_SHEET, field, "Empty item in comma-separated list (e.g. trailing comma)",
                "Empty item removed", example=f"{sku}: '{raw}'")
    out: list[str] = []
    for p in parts:
        if p and p.lower() not in {o.lower() for o in out}:
            out.append(p)
    if not out:
        log.add(VENDOR_SHEET, field, "Missing value", "Attribute omitted from listing", example=sku, severity="warning")
    return out


def _parse_bool(value) -> bool | None:
    v = _text(value).lower()
    return True if v in _YES else False if v in _NO else None


def _same_dims(a: Dimensions, b: Dimensions) -> bool:
    return all(abs(x - y) < 0.01 for x, y in [(a.length, b.length), (a.width, b.width), (a.height, b.height)])


def _positive(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def _text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()
