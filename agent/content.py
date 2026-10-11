"""Listing content per SKU.

Split of responsibility:
  - attributes  -> built deterministically from cleaned vendor facts (cannot hallucinate)
  - title, description, 5 key features -> LLM, constrained to the fact sheet
  - QA          -> deterministic checks (grounded numbers, unsupported claims, qualifiers kept,
                   no placeholders/truncation, length) -> issues fed back to the LLM for a rewrite
"""
from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field

from .llm import LLM, LLMError
from .models import Product

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


class KeyFeature(BaseModel):
    label: str = Field(description="2-4 word benefit label in Title Case, e.g. 'Power Recline and Headrest'")
    text: str = Field(description="One complete sentence (max ~30 words) ending with a period")


class ListingCopy(BaseModel):
    title: str = Field(description="Marketplace title, 70-150 characters")
    description: str = Field(description="Product description, 90-160 words, one or two paragraphs")
    key_features: list[KeyFeature] = Field(min_length=5, max_length=5)


class ContentResult(BaseModel):
    sku: str
    title: str = ""
    description: str = ""
    key_features: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    qa_status: str = "not_run"  # passed | passed_with_warnings | failed | error
    qa_issues: list[str] = Field(default_factory=list)
    attempts: int = 0
    model: str = ""


SYSTEM = """You write marketplace-ready product listings for Benjara, a US furniture brand sold on its own storefront
and on marketplaces (Amazon, Wayfair, Walmart). Tone: confident, warm, specific, plain American English. No hype.

Hard rules:
- Use ONLY information in the FACTS. Do not add specs, numbers, materials, certifications, warranties, care
  instructions, room sizes, TV sizes, or use cases that the facts do not support.
- Vendor bullets have grammar errors and filler ("skin loving smoothness", "serve plush comfort"): rewrite them cleanly.
- Keep material qualifiers exactly: "Leather Aire" is not "leather"; "faux marble" is not "marble";
  "marble glass" is not "marble".
- Every sentence must be complete. No placeholders, no exclamation marks, no ALL-CAPS words, no emojis.
- Never mention other collections or "check out our ... collection".
"""

GUIDE = """Write the listing.

TITLE: "<Collection> <Product Type> with/, <2-3 most distinctive features>, <Colour> <Key Material>".
  Title Case, 70-150 characters, must include the collection name and the primary colour exactly as given.
  Example shape (different product): "Kiri Swivel Accent Chair with Channel Tufting, Metal Base, Cream Boucle Fabric".
  Colour variants of the same model must be distinguishable from the title alone.
DESCRIPTION: 90-160 words. Open with what the product is and who/where it suits, then the most important
  functional and design details. If you mention dimensions, copy them exactly in the form 41"L x 37"W x 41"H.
KEY FEATURES: exactly 5, ordered by purchase importance (core function, comfort/storage, build/materials,
  design/finish, convenience/safety). Each: a short label + one sentence that adds detail rather than
  repeating the label's words. Do not repeat the same point twice.
"""

RISKY_CLAIMS = [
    "warranty", "guarantee", "waterproof", "water-resistant", "scratch-resistant", "stain-resistant",
    "eco-friendly", "non-toxic", "fsc", "certified", "handmade", "handcrafted", "solid wood", "genuine leather",
    "real leather", "top-grain", "top grain", "100%", "free shipping", "best-selling", "bestseller", "#1",
    "award", "lifetime", "commercial grade", "pet-friendly", "kid-safe", "child-safe", "easy assembly",
    "tool-free", "no tools", "assembly-free", "fully assembled", "energy efficient", "remote control",
]
# material phrase in facts -> bare word that must not appear on its own in the copy
QUALIFIED_MATERIALS = {
    "leather aire": "leather", "faux leather": "leather", "faux marble": "marble", "marble glass": "marble",
}
QUALIFIER_ALLOWED = re.compile(r"(faux|marble[- ]look|marble[- ]effect|marble[- ]inspired|leather[- ]look)", re.I)
PLACEHOLDER_RE = re.compile(r"lorem|ipsum|\bTBD\b|\bTODO\b|[\[\]{}<>]|\bN/A\b|\bXX+\b|\binsert\b", re.I)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}  # "Three piece" -> 3


def generate_content(product: Product, llm: LLM) -> ContentResult:
    result = ContentResult(sku=product.sku, attributes=build_attributes(product))
    facts = product.fact_sheet()
    facts_json = json.dumps(facts, indent=1)
    best: tuple[int, ListingCopy, list[str], str] | None = None
    feedback = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = f"{GUIDE}\nFACTS (JSON):\n{facts_json}{feedback}"
        try:
            out = llm.structured(system=SYSTEM, prompt=prompt, schema=ListingCopy, namespace="content", temperature=0.5)
        except LLMError as exc:
            log.error("Content generation failed for %s: %s", product.sku, exc)
            result.qa_issues.append(f"LLM error: {exc}")
            break
        copy: ListingCopy = out.value
        issues = qa_issues(copy, product)
        result.attempts = attempt
        if best is None or len(issues) < len(best[2]):
            best = (attempt, copy, issues, f"{out.provider}:{out.model}")
        if not issues:
            break
        log.info("%s attempt %d QA issues: %s", product.sku, attempt, issues)
        feedback = (
            "\n\nYour previous draft failed these checks. Rewrite the whole listing and fix every one:\n- "
            + "\n- ".join(issues)
            + f"\n\nPREVIOUS DRAFT:\n{copy.model_dump_json(indent=1)}"
        )

    if best is None:
        result.qa_status = "error"
        return result
    _, copy, issues, model = best
    result.title = copy.title.strip()
    result.description = copy.description.strip()
    result.key_features = [f"{f.label.strip()}: {f.text.strip()}" for f in copy.key_features]
    result.qa_issues = issues
    result.qa_status = "passed" if not issues else "passed_with_warnings" if _minor(issues) else "failed"
    result.model = model
    return result


# --- QA -------------------------------------------------------------------------------------


def qa_issues(copy: ListingCopy, p: Product) -> list[str]:
    issues: list[str] = []
    features = [f"{f.label} {f.text}" for f in copy.key_features]
    all_text = " ".join([copy.title, copy.description, *features])
    source = _source_text(p).lower()

    if not 60 <= len(copy.title) <= 160:
        issues.append(f"Title is {len(copy.title)} characters; keep it between 70 and 150.")
    if p.collection and p.collection.lower() not in copy.title.lower():
        issues.append(f"Title must include the collection name '{p.collection}'.")
    if p.colors and p.colors[0].lower() not in copy.title.lower():
        issues.append(f"Title must include the primary colour '{p.colors[0]}'.")
    if copy.title.rstrip()[-1:] in ".,;:-":
        issues.append("Title must not end with punctuation.")

    words = len(copy.description.split())
    if not 70 <= words <= 190:
        issues.append(f"Description is {words} words; keep it between 90 and 160.")
    if not copy.description.rstrip().endswith((".", "?")):
        issues.append("Description must end with a complete sentence.")
    for f in copy.key_features:
        if not f.text.rstrip().endswith("."):
            issues.append(f"Feature '{f.label}' must be one complete sentence ending with a period.")
        if len(f.text) > 260:
            issues.append(f"Feature '{f.label}' is too long ({len(f.text)} chars); keep it to one sentence.")
        if f.text.lower().startswith(f.label.lower().strip()):
            issues.append(f"Feature '{f.label}' repeats its label; start the sentence differently.")
    labels = [f.label.strip().lower() for f in copy.key_features]
    if len(set(labels)) < len(labels):
        issues.append("Feature labels repeat; each of the 5 features must cover a different point.")

    known = {_norm_num(n) for n in NUMBER_RE.findall(source)}
    known |= {str(v) for w, v in NUMBER_WORDS.items() if re.search(rf"\b{w}\b", source)}
    unknown = sorted({n for n in NUMBER_RE.findall(all_text) if _norm_num(n) not in known})
    if unknown:
        issues.append(f"Numbers not found in the facts: {', '.join(unknown)}. Remove them or use the exact fact values.")

    claims = [c for c in RISKY_CLAIMS if c in all_text.lower() and c not in source]
    if claims:
        issues.append(f"Unsupported claims not in the facts: {', '.join(claims)}.")

    for phrase, bare in QUALIFIED_MATERIALS.items():
        if phrase in source and bare not in source.replace(phrase, ""):
            stripped = QUALIFIER_ALLOWED.sub("", all_text.lower().replace(phrase, ""))
            if re.search(rf"\b{bare}\b", stripped):
                issues.append(f"Write '{phrase}' (or a clear 'faux'/'-look' qualifier), never plain '{bare}'.")

    if PLACEHOLDER_RE.search(all_text):
        issues.append("Remove placeholder-like text or brackets.")
    if "!" in all_text:
        issues.append("Remove exclamation marks.")
    if re.search(r"check out our", all_text, re.I):
        issues.append("Remove cross-sell text ('check out our ... collection').")
    caps = [w for w in re.findall(r"\b[A-Z]{4,}\b", all_text) if w not in {"USB", "LED", "MDF", "USB-C", "USB-A"}]
    if caps:
        issues.append(f"Avoid ALL-CAPS words: {', '.join(sorted(set(caps)))}.")
    return issues


def _minor(issues: list[str]) -> bool:
    """Length/format issues are warnings; grounding problems (numbers, claims, materials) are failures."""
    serious = ("Numbers not found", "Unsupported claims", "never plain", "placeholder")
    return not any(s in i for i in issues for s in serious)


def _source_text(p: Product) -> str:
    return json.dumps(p.fact_sheet())


def _norm_num(n: str) -> str:
    return f"{float(n):g}"


# --- deterministic attributes ---------------------------------------------------------------


def build_attributes(p: Product) -> dict[str, str]:
    attrs: dict[str, str] = {}
    if p.product_type:
        attrs["Product Type"] = _title_case(p.product_type)
    if p.collection:
        attrs["Collection"] = p.collection
    if p.brand:
        attrs["Brand"] = p.brand
    if p.colors:
        attrs["Colour"] = ", ".join(p.colors)
    if p.materials:
        attrs["Material"] = ", ".join(p.materials)
    if p.style:
        attrs["Style"] = p.style.title()
    attrs["Assembly"] = _assembly(p)
    if p.dimensions:
        attrs["Dimensions (L x W x H)"] = p.dimensions.label()
    extra = [_dimension_extra(d) for d in p.dimension_details]
    if extra := [d for d in extra if d]:
        attrs["Dimension Details"] = " | ".join(extra)
    if p.product_weight_lb:
        attrs["Product Weight"] = f"{p.product_weight_lb:g} lb"
    if p.weight_capacity:
        attrs["Weight Capacity"] = _capacity(p.weight_capacity)
    if p.included:
        attrs["What's Included"] = p.included
    if p.cartons:
        weights = ", ".join(f"{c.weight_lb:g} lb" for c in p.cartons)
        attrs["Shipping"] = f"{len(p.cartons)} carton{'s' if len(p.cartons) > 1 else ''} ({weights})"
    return {k: v for k, v in attrs.items() if v}


def _dimension_extra(detail: str) -> str:
    """Drop the 'Overall: ...' part already shown under Dimensions; keep component and seat measurements."""
    parts = [x for x in detail.split("; ") if not x.startswith("Overall: ")]
    return "; ".join(parts)


def _assembly(p: Product) -> str:
    base = {True: "Required", False: "Not required", None: ""}[p.assembly_required]
    note = p.assembly_note or ""
    generic = re.fullmatch(r"(no )?assembly required\.?", note, re.I)
    if note and not generic:
        return f"{base} (vendor note: {note[0].lower() + note[1:]})" if base else note
    return base or "Not specified"


def _capacity(text: str) -> str:
    text = re.sub(r"(\d)\s*lbs?\b", r"\1 lb", text)
    return re.sub(r"\s+comfortably\.?$", "", text).rstrip(".")


def _title_case(s: str) -> str:
    return " ".join(w if w.isupper() else w.capitalize() for w in s.split())
