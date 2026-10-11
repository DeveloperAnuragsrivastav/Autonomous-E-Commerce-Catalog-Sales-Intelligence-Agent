"""Catalog agent: one command from the vendor/sales workbook to all outputs.

    python main.py --input data/test_data.xlsx
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

from agent.cache import DiskCache
from agent.config import load_settings
from agent.graph import RunContext, build_graph


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate listings, lifestyle images, competitor benchmarks and sales analysis from one workbook.")
    p.add_argument("--input", required=True, type=Path, help="Path to the vendor + sales .xlsx workbook")
    p.add_argument("--skus", help="Comma-separated vendor SKUs to process (default: all)")
    p.add_argument("--image-provider", choices=["auto", "gemini", "composite"], default="auto",
                   help="auto = Gemini image edit, falling back to the local composite renderer")
    p.add_argument("--image-limit", type=int, help="Only generate images for the first N SKUs (cost control)")
    p.add_argument("--skip-images", action="store_true")
    p.add_argument("--skip-competitors", action="store_true")
    p.add_argument("--competitor-limit", type=int, default=0,
                   help="Max models to benchmark (default 0 = every model; colour variants share one benchmark)")
    p.add_argument("--competitor-skus", help="Comma-separated SKUs to benchmark (default: one per model)")
    p.add_argument("--competitors-per-sku", type=int, default=3)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--images-dir", type=Path)
    p.add_argument("--workers", type=int, default=4, help="Parallel LLM calls for content generation")
    p.add_argument("--no-cache", action="store_true", help="Ignore cached LLM/search/image results")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "groq"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message=".*automatic function calling.*")

    split = lambda s: tuple(x.strip() for x in s.split(",") if x.strip()) if s else None  # noqa: E731
    settings = load_settings(
        input_path=args.input.resolve(),
        skus=split(args.skus),
        image_provider=args.image_provider,
        image_limit=args.image_limit,
        skip_images=args.skip_images,
        skip_competitors=args.skip_competitors,
        competitor_limit=args.competitor_limit,
        competitor_skus=split(args.competitor_skus),
        competitors_per_sku=args.competitors_per_sku,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        images_dir=args.images_dir.resolve() if args.images_dir else None,
        max_workers=args.workers,
        use_cache=not args.no_cache,
    )
    ctx = RunContext(settings=settings, cache=DiskCache(settings.cache_dir, enabled=settings.use_cache))

    t0 = time.monotonic()
    final = build_graph(ctx).invoke({"steps": []})
    if final.get("fatal_error"):
        logging.error("Run stopped: %s", final["fatal_error"])
        return 1

    print(f"\nDone in {time.monotonic() - t0:.0f}s")
    for s in final.get("steps", []):
        print(f"  {s['status']:>7}  {s['step']:<22} {s['seconds']:>6.1f}s  {s['detail']}")
    print("\nOutputs:")
    for name, path in final.get("outputs", {}).items():
        print(f"  {name:<18} {path}")
    failed = [s for s in final.get("steps", []) if s["status"] == "failed"]
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
