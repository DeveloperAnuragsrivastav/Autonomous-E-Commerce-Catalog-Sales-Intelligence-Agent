"""Read a finished run's output files into plain dicts for the templates."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DOWNLOADS = {
    "catalog_agent_results.xlsx": ("Results workbook", "Content, competitors, sales and data quality in 20 sheets. The main deliverable."),
    "report.html": ("HTML report", "One readable page with every listing next to its image."),
    "data_quality.md": ("Data quality report", "Every anomaly found in the input and the action taken."),
    "content.json": ("Content (JSON)", "Titles, descriptions, features, attributes and QA per SKU."),
    "competitors.json": ("Competitors (JSON)", "Benchmarks, competitor rows and storefront prices."),
    "sales_analysis.json": ("Sales analysis (JSON)", "Metrics, tables and insights."),
    "run_manifest.json": ("Run manifest", "Input hash, settings, step timing and models used."),
}


@dataclass
class RunResults:
    content: list[dict] = field(default_factory=list)
    products: dict[str, dict] = field(default_factory=dict)
    images: dict[str, dict] = field(default_factory=dict)
    competitors: dict = field(default_factory=dict)
    sales: dict | None = None
    quality: list[dict] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)
    files: list[dict] = field(default_factory=list)

    @property
    def has_output(self) -> bool:
        return bool(self.manifest)

    def our_price(self, sku: str) -> dict | None:
        return next((p for p in self.competitors.get("our_prices", []) if p["sku"] == sku), None)

    def benchmark_for(self, sku: str) -> tuple[dict | None, list[dict]]:
        summary = next((s for s in self.competitors.get("summaries", []) if sku in s["covers_skus"]), None)
        rows = [r for r in self.competitors.get("rows", []) if summary and r["sku"] == summary["sku"]]
        return summary, rows


def load_results(run_dir: Path) -> RunResults:
    out = run_dir / "output"
    r = RunResults()
    r.content = _json(out / "content.json", [])
    r.products = {p["sku"]: p for p in _json(out / "products.json", [])}
    r.images = {i["sku"]: i for i in _json(out / "images.json", [])}
    if not r.images:  # older runs: per-image sidecars
        for meta in sorted((run_dir / "images").glob("*.json")):
            data = _json(meta, {})
            if data.get("sku"):
                r.images[data["sku"]] = data
    if not r.products:  # older runs: derive minimal product facts from content
        r.products = {c["sku"]: {"sku": c["sku"], "product_type": c["attributes"].get("Product Type"),
                                 "collection": c["attributes"].get("Collection"), "image_url": r.images.get(c["sku"], {}).get("source_url")}
                      for c in r.content}
    r.competitors = _json(out / "competitors.json", {})
    if "our_prices" not in r.competitors:
        r.competitors["our_prices"] = [s["our_price"] for s in r.competitors.get("summaries", [])]
    r.sales = _json(out / "sales_analysis.json", None)
    r.quality = _json(out / "data_quality.json", [])
    r.manifest = _json(out / "run_manifest.json", {})
    r.files = [
        {"name": name, "label": label, "description": desc, "size": (out / name).stat().st_size}
        for name, (label, desc) in DOWNLOADS.items() if (out / name).exists()
    ]
    return r


def _json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default
