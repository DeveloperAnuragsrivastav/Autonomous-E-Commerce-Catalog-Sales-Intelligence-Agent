"""Sales sheet -> one clean line-level DataFrame with an explicit `exclusion_reason` per line.

Nothing is silently dropped: every line stays in the frame, and lines that must not count as
sales carry the reason. Sales metrics use `counts_as_sale`; the report shows the exclusions.
"""
from __future__ import annotations

import pandas as pd

from .config import SALES_SHEET
from .quality import QualityLog

# Status vocabulary -> bucket. Unknown statuses are excluded and flagged rather than guessed.
STATUS_BUCKETS = {
    "processed": "fulfilled",
    "delivered": "fulfilled",
    "shipped": "in_transit",
    "oj shipping": "in_transit",
    "pending": "pending",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}
SOLD_BUCKETS = {"fulfilled", "in_transit"}

CARRIERS = {"fedx": "FedEx", "fedex": "FedEx", "upsn": "UPS", "ups": "UPS", "ltl": "LTL Freight", "usps": "USPS"}
UNMAPPED_CARRIER = "Unmapped"

# Columns renamed to what they actually contain (see README: `Vendor` holds the fulfilment
# route and `vendorType` holds the supplier name).
RENAMES = {
    "orderDate": "order_date", "orderId": "order_id", "invoiceNumber": "invoice_number",
    "customerReferenceNumber": "customer_reference", "itemName": "item_name", "Masked SKU": "sku",
    "quantity": "quantity", "price": "unit_price", "amount": "amount", "Customer Name": "channel",
    "orderStatus": "status_raw", "shippingCarrier": "carrier_raw", "Vendor": "fulfilment_route",
    "vendorType": "supplier",
}


def clean_sales(raw: pd.DataFrame, log: QualityLog) -> pd.DataFrame:
    df = raw.rename(columns=RENAMES).copy()
    n = len(df)

    exact_dupes = df.duplicated()
    if exact_dupes.any():
        log.add(SALES_SHEET, "all columns", "Exact duplicate lines", "Dropped", count=int(exact_dupes.sum()),
                severity="warning")
        df = df[~exact_dupes].copy()

    df["sku"] = _clean_sku(df, log)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for col in ("quantity", "unit_price", "amount"):
        bad = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
        log.add(SALES_SHEET, col, "Non-numeric value", "Treated as missing; line excluded", count=int(bad.sum()),
                severity="warning")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["channel"] = _normalise_labels(df["channel"], "Customer Name", log)
    df["supplier"] = _normalise_labels(df["supplier"], "vendorType", log)
    df["fulfilment_route"] = _normalise_labels(df["fulfilment_route"], "Vendor", log)
    log.add(SALES_SHEET, "Vendor / vendorType",
            "Column names do not match contents: 'Vendor' holds the fulfilment route (WareHouse, Import, ...) "
            "and 'vendorType' holds the supplier name",
            "Renamed to fulfilment_route and supplier", count=1)

    df["status"] = _status_bucket(df["status_raw"], log)
    df["carrier"] = _carrier(df["carrier_raw"], log)

    mismatch = (df["unit_price"] * df["quantity"] - df["amount"]).abs() > 0.01
    log.add(SALES_SHEET, "amount", "amount != price x quantity", "Kept invoiced amount; flagged",
            count=int(mismatch.sum()), severity="warning")

    item_diff = df["item_name"].astype(str).str.strip() != df["sku"]
    log.add(SALES_SHEET, "itemName", "itemName differs from Masked SKU (internal item codes)",
            "Masked SKU used as the product key; itemName kept for reference", count=int(item_diff.sum()),
            examples=[f"{i} -> {s}" for i, s in df.loc[item_diff, ["item_name", "sku"]].head(3).itertuples(index=False)])

    df["exclusion_reason"] = _exclusion_reasons(df)
    df["counts_as_sale"] = df["exclusion_reason"].eq("")
    for reason, count in df.loc[~df["counts_as_sale"], "exclusion_reason"].value_counts().items():
        log.add(SALES_SHEET, "line", f"Excluded from sales: {reason}", "Not counted in revenue or units",
                count=int(count))

    _log_date_coverage(df, log)
    assert len(df) == n - int(exact_dupes.sum())
    return df


def _clean_sku(df: pd.DataFrame, log: QualityLog) -> pd.Series:
    raw = df["sku"]
    numeric = raw.map(lambda v: isinstance(v, (int, float)) and not pd.isna(v))
    log.add(SALES_SHEET, "Masked SKU", "SKU stored as a number", "Converted to text so it joins consistently",
            count=int(numeric.sum()), examples=[str(v) for v in raw[numeric].head(3)])
    sku = raw.map(_sku_text)
    fallback = sku.eq("") & df["item_name"].notna()
    log.add(SALES_SHEET, "Masked SKU", "Missing SKU", "Fell back to itemName", count=int(fallback.sum()),
            severity="warning")
    return sku.where(~fallback, df["item_name"].map(_sku_text))


def _sku_text(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def _normalise_labels(series: pd.Series, field: str, log: QualityLog) -> pd.Series:
    """Trim whitespace and merge case variants ('Merrick home' / 'Merrick Home') to the most common spelling."""
    stripped = series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    changed = (stripped != series.astype("string")).fillna(False)
    log.add(SALES_SHEET, field, "Leading/trailing or repeated whitespace", "Trimmed",
            count=int(changed.sum()), examples=sorted({repr(v) for v in series[changed].unique()})[:3])
    canonical = (
        stripped.dropna().to_frame("v").assign(k=lambda d: d.v.str.casefold())
        .groupby("k")["v"].agg(lambda s: s.value_counts().index[0])
    )
    variants = stripped.dropna().groupby(stripped.dropna().str.casefold()).nunique()
    merged = variants[variants > 1]
    if len(merged):
        log.add(SALES_SHEET, field, "Same name with different capitalisation", "Merged to most common spelling",
                count=len(merged), examples=[canonical[k] for k in merged.index[:3]])
    out = stripped.str.casefold().map(canonical)
    return out.fillna("Unknown")


def _status_bucket(status: pd.Series, log: QualityLog) -> pd.Series:
    key = status.astype("string").str.strip().str.casefold()
    bucket = key.map(STATUS_BUCKETS)
    unknown = bucket.isna()
    if unknown.any():
        log.add(SALES_SHEET, "orderStatus", "Unrecognised order status", "Excluded from sales and listed for review",
                count=int(unknown.sum()), examples=sorted(status[unknown].astype(str).unique())[:5], severity="warning")
    oj = key.eq("oj shipping")
    log.add(SALES_SHEET, "orderStatus",
            "'OJ Shipping' status (not described in the brief; all FedEx, looks like an order in transit)",
            "Counted as a sale (order accepted and shipping)", count=int(oj.sum()))
    return bucket.fillna("unknown")


def _carrier(carrier: pd.Series, log: QualityLog) -> pd.Series:
    key = carrier.astype("string").str.strip().str.casefold()
    mapped = key.map(CARRIERS)
    unmapped = mapped.isna()
    values = sorted(carrier[unmapped].astype(str).unique())
    log.add(SALES_SHEET, "shippingCarrier", "Carrier not mapped / unknown carrier value",
            f"Grouped as '{UNMAPPED_CARRIER}'; lines still count as sales (carrier does not affect revenue)",
            count=int(unmapped.sum()), examples=values[:5], severity="warning")
    return mapped.fillna(UNMAPPED_CARRIER)


def _exclusion_reasons(df: pd.DataFrame) -> pd.Series:
    """First matching rule wins, so each excluded line is counted under exactly one reason."""
    rules = [
        (df["status"].eq("cancelled"), "cancelled order"),
        (df["status"].eq("pending"), "pending order (not yet confirmed)"),
        (df["status"].eq("unknown"), "unrecognised status"),
        (df["sku"].eq(""), "missing SKU"),
        (df["order_date"].isna(), "invalid order date"),
        (df["quantity"].isna() | (df["quantity"] <= 0), "zero or missing quantity"),
        (df["amount"].isna() | (df["amount"] <= 0) | (df["unit_price"] <= 0),
         "zero-value line (price 0: likely replacement or sample)"),
    ]
    reason = pd.Series("", index=df.index)
    for mask, label in reversed(rules):
        reason = reason.mask(mask.fillna(True), label)
    return reason


def _log_date_coverage(df: pd.DataFrame, log: QualityLog) -> None:
    dates = df["order_date"].dropna()
    if dates.empty:
        return
    by_month = dates.groupby(dates.dt.to_period("M")).agg(["min", "max", "count"])
    for period, r in by_month.iterrows():
        days_covered = r["max"].day - r["min"].day + 1
        if days_covered < period.days_in_month - 1:
            log.add(SALES_SHEET, "orderDate",
                    f"{period.strftime('%b %Y')} is partial: {r['min']:%d %b}-{r['max']:%d %b} "
                    f"({days_covered} of {period.days_in_month} days, {r['count']} lines)",
                    "Kept in totals; excluded from month-over-month comparisons", count=1)


def sold_lines(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["counts_as_sale"]]


def full_months(df: pd.DataFrame) -> list[pd.Period]:
    """Months with (near-)complete coverage, for fair month-over-month comparison."""
    dates = df["order_date"].dropna()
    spans = dates.groupby(dates.dt.to_period("M")).agg(["min", "max"])
    return [p for p, r in spans.iterrows() if r["max"].day - r["min"].day + 1 >= p.days_in_month - 1]

