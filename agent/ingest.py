"""Read the workbook and validate it has what the pipeline needs. No cleaning happens here."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import SALES_SHEET, VENDOR_SHEET

VENDOR_REQUIRED = ["SKU", "Category", "Color", "Material Used", "Assembly Required", "Brand", "Image URL 1"]
SALES_REQUIRED = [
    "orderDate", "orderId", "Masked SKU", "itemName", "quantity", "price", "amount",
    "Customer Name", "orderStatus", "shippingCarrier", "Vendor", "vendorType",
]


class InputError(Exception):
    """The input file cannot be processed at all (missing file, sheet or required column)."""


def load_workbook(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise InputError(f"Input file not found: {path}")
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as exc:  # corrupt / not an xlsx
        raise InputError(f"Could not read {path.name} as an Excel workbook: {exc}") from exc

    by_name = {name.strip().lower(): df for name, df in sheets.items()}
    vendor = _sheet(by_name, VENDOR_SHEET, list(sheets))
    sales = _sheet(by_name, SALES_SHEET, list(sheets))
    for df in (vendor, sales):
        df.columns = [str(c).strip() for c in df.columns]

    _require(vendor, VENDOR_REQUIRED, VENDOR_SHEET)
    if not any(c.lower().startswith("bullet") for c in vendor.columns):
        raise InputError(f"Sheet '{VENDOR_SHEET}' has no 'Bullet N' columns")
    _require(sales, SALES_REQUIRED, SALES_SHEET)
    return vendor, sales


def _sheet(by_name: dict[str, pd.DataFrame], wanted: str, available: list[str]) -> pd.DataFrame:
    if wanted not in by_name:
        raise InputError(f"Sheet '{wanted}' not found. Sheets present: {available}")
    return by_name[wanted]


def _require(df: pd.DataFrame, columns: list[str], sheet: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise InputError(f"Sheet '{sheet}' is missing required columns: {missing}")
