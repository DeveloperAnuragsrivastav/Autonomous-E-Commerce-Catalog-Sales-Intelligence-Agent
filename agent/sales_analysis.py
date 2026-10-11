"""Sales analysis: deterministic metrics (pandas) + LLM-written insights grounded in those metrics.

The LLM never computes numbers. It receives a compact fact sheet, and every number it writes
is checked against that sheet; ungrounded output is retried once, then replaced by rule-based insights.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, Field

from .grounding import ungrounded
from .llm import LLM, LLMError
from .sales_clean import SOLD_BUCKETS, full_months, sold_lines

log = logging.getLogger(__name__)

TOP_N = 10
LOW_MOVING_MAX_UNITS = 1   # sold at most this many units across the whole period
STALLED_MIN_UNITS = 3      # sold at least this many in the first full month, none in the last


class Insight(BaseModel):
    headline: str = Field(description="At most 12 words, states the finding")
    detail: str = Field(description="1-2 sentences with the supporting numbers, copied from the facts")
    recommendation: str = Field(description="One concrete action for the business")


class InsightSet(BaseModel):
    insights: list[Insight] = Field(min_length=3, max_length=5)


@dataclass
class SalesReport:
    summary: dict
    monthly: pd.DataFrame
    weekly: pd.DataFrame
    top_by_revenue: pd.DataFrame
    top_by_units: pd.DataFrame
    low_moving: pd.DataFrame
    non_moving: pd.DataFrame
    stalled: pd.DataFrame
    by_channel: pd.DataFrame
    by_route: pd.DataFrame
    by_carrier: pd.DataFrame
    by_supplier: pd.DataFrame
    exclusions: pd.DataFrame
    facts: dict
    insights: list[Insight] = field(default_factory=list)
    insights_source: str = ""
    treatment_notes: list[str] = field(default_factory=list)


def analyse_sales(df: pd.DataFrame) -> SalesReport:
    sold = sold_lines(df)
    per_sku = _per_sku(df, sold)
    months = full_months(df)

    revenue = float(sold["amount"].sum())
    orders = int(sold["order_id"].nunique())
    summary = {
        "period_start": df["order_date"].min().date().isoformat(),
        "period_end": df["order_date"].max().date().isoformat(),
        "net_sales": round(revenue, 2),
        "units_sold": int(sold["quantity"].sum()),
        "orders": orders,
        "order_lines": int(len(sold)),
        "skus_sold": int(sold["sku"].nunique()),
        "skus_in_file": int(df["sku"].nunique()),
        "avg_order_value": round(revenue / orders, 2) if orders else 0.0,
        "avg_unit_price": round(revenue / sold["quantity"].sum(), 2) if len(sold) else 0.0,
        "lines_in_file": int(len(df)),
        "lines_excluded": int((~df["counts_as_sale"]).sum()),
        "pending_value": round(float(df.loc[df["status"].eq("pending"), "amount"].sum()), 2),
        "cancelled_lines": int(df["status"].eq("cancelled").sum()),
        "cancelled_value_recorded": round(float(df.loc[df["status"].eq("cancelled"), "amount"].sum()), 2),
    }

    report = SalesReport(
        summary=summary,
        monthly=_monthly(df, sold, months),
        weekly=_weekly(df, sold),
        top_by_revenue=per_sku.sort_values(["revenue", "units"], ascending=False).head(TOP_N).reset_index(),
        top_by_units=per_sku.sort_values(["units", "revenue"], ascending=False).head(TOP_N).reset_index(),
        low_moving=_low_moving(per_sku),
        non_moving=_non_moving(df, per_sku),
        stalled=_stalled(sold, months),
        by_channel=_by_channel(df, sold),
        by_route=_share(sold, "fulfilment_route"),
        by_carrier=_share(sold, "carrier"),
        by_supplier=_share(sold, "supplier").head(15),
        exclusions=_exclusions(df),
        facts={},
        treatment_notes=_treatment_notes(df, months),
    )
    report.facts = _facts(report, per_sku, months)
    return report


# --- metrics ------------------------------------------------------------------------------


def _per_sku(df: pd.DataFrame, sold: pd.DataFrame) -> pd.DataFrame:
    agg = sold.groupby("sku").agg(
        units=("quantity", "sum"), revenue=("amount", "sum"), orders=("order_id", "nunique"),
        avg_price=("unit_price", "mean"), last_sold=("order_date", "max"),
    )
    all_skus = pd.Index(df["sku"].unique(), name="sku")
    agg = agg.reindex(all_skus).fillna({"units": 0, "revenue": 0.0, "orders": 0})
    agg["units"] = agg["units"].astype(int)
    agg["orders"] = agg["orders"].astype(int)
    agg["revenue"] = agg["revenue"].round(2)
    agg["avg_price"] = agg["avg_price"].round(2)
    total = agg["revenue"].sum()
    agg["revenue_share_pct"] = (agg["revenue"] / total * 100).round(2) if total else 0.0
    return agg


def _monthly(df: pd.DataFrame, sold: pd.DataFrame, months: list[pd.Period]) -> pd.DataFrame:
    period = sold["order_date"].dt.to_period("M")
    m = sold.groupby(period).agg(net_sales=("amount", "sum"), units=("quantity", "sum"), orders=("order_id", "nunique"))
    cancelled = df[df["status"].eq("cancelled")].groupby(df["order_date"].dt.to_period("M")).size()
    m["cancelled_lines"] = cancelled.reindex(m.index).fillna(0).astype(int)
    days = df.groupby(df["order_date"].dt.to_period("M"))["order_date"].agg(lambda d: d.dt.date.nunique())
    m["days_with_orders"] = days.reindex(m.index)
    m["complete_month"] = [p in months for p in m.index]
    m["net_sales"] = m["net_sales"].round(2)
    m.index = m.index.strftime("%Y-%m")
    return m.rename_axis("month").reset_index()


def _weekly(df: pd.DataFrame, sold: pd.DataFrame) -> pd.DataFrame:
    """Monday-start weeks; a week the data only partly covers is flagged so charts can mark it."""
    first, last = df["order_date"].min(), df["order_date"].max()
    week = sold["order_date"].dt.to_period("W-SUN")
    w = sold.groupby(week).agg(net_sales=("amount", "sum"), units=("quantity", "sum"), orders=("order_id", "nunique"))
    w["week_start"] = [p.start_time.date().isoformat() for p in w.index]
    w["week_end"] = [p.end_time.date().isoformat() for p in w.index]
    w["days_covered"] = [
        (min(p.end_time.normalize(), last) - max(p.start_time, first)).days + 1 for p in w.index
    ]
    w["partial"] = w["days_covered"] < 7
    w["net_sales"] = w["net_sales"].round(2)
    return w.reset_index(drop=True)


def _low_moving(per_sku: pd.DataFrame) -> pd.DataFrame:
    low = per_sku[(per_sku["units"] > 0) & (per_sku["units"] <= LOW_MOVING_MAX_UNITS)]
    return low.sort_values(["revenue", "last_sold"]).reset_index()


def _non_moving(df: pd.DataFrame, per_sku: pd.DataFrame) -> pd.DataFrame:
    """SKUs that were ordered but never counted as sold (every line cancelled, pending or zero-value)."""
    skus = per_sku.index[per_sku["units"] == 0]
    lines = df[df["sku"].isin(skus)]
    out = lines.groupby("sku").agg(
        order_lines=("order_id", "size"),
        why=("exclusion_reason", lambda r: ", ".join(f"{k} x{v}" for k, v in r.value_counts().items())),
        last_order=("order_date", "max"),
        lost_value_recorded=("amount", "sum"),
    )
    return out.sort_values(["order_lines", "lost_value_recorded"], ascending=False).reset_index()


def _stalled(sold: pd.DataFrame, months: list[pd.Period]) -> pd.DataFrame:
    """Sold steadily in the first full month, nothing in the last full month."""
    if len(months) < 2:
        return pd.DataFrame(columns=["sku", "units_first_month", "units_last_month"])
    first, last = months[0], months[-1]
    period = sold["order_date"].dt.to_period("M")
    a = sold[period == first].groupby("sku")["quantity"].sum()
    b = sold[period == last].groupby("sku")["quantity"].sum()
    stalled = a[(a >= STALLED_MIN_UNITS) & ~a.index.isin(b.index)]
    return (
        stalled.rename("units_first_month").to_frame().assign(units_last_month=0)
        .sort_values("units_first_month", ascending=False).reset_index()
    )


def _by_channel(df: pd.DataFrame, sold: pd.DataFrame) -> pd.DataFrame:
    total = sold["amount"].sum()
    g = df.groupby("channel")
    out = pd.DataFrame({
        "net_sales": sold.groupby("channel")["amount"].sum(),
        "orders": sold.groupby("channel")["order_id"].nunique(),
        "lines_total": g.size(),
        "cancel_rate_pct": g["status"].apply(lambda s: (s == "cancelled").mean() * 100),
        "unmapped_carrier_pct": g["carrier"].apply(lambda s: (s == "Unmapped").mean() * 100),
    }).fillna(0)
    out["revenue_share_pct"] = out["net_sales"] / total * 100
    return out.sort_values("net_sales", ascending=False).round(2).rename_axis("channel").reset_index()


def _share(sold: pd.DataFrame, col: str) -> pd.DataFrame:
    total = sold["amount"].sum()
    out = sold.groupby(col).agg(net_sales=("amount", "sum"), units=("quantity", "sum"), lines=("amount", "size"))
    out["revenue_share_pct"] = out["net_sales"] / total * 100
    out["avg_line_value"] = out["net_sales"] / out["lines"]
    return out.sort_values("net_sales", ascending=False).round(2).reset_index()


def _exclusions(df: pd.DataFrame) -> pd.DataFrame:
    ex = df[~df["counts_as_sale"]]
    out = ex.groupby("exclusion_reason").agg(lines=("sku", "size"), units=("quantity", "sum"), value_recorded=("amount", "sum"))
    return out.round(2).sort_values("lines", ascending=False).reset_index()


def _treatment_notes(df: pd.DataFrame, months: list[pd.Period]) -> list[str]:
    status_counts = df["status_raw"].value_counts().to_dict()
    sold_statuses = sorted({s for s, b in zip(df["status_raw"], df["status"]) if b in SOLD_BUCKETS})
    partial = sorted(set(df["order_date"].dt.to_period("M").dropna()) - set(months))
    zeroed = int((df["status"].eq("cancelled") & df["quantity"].le(0)).sum())
    notes = [
        f"Counted as sales: statuses {', '.join(sold_statuses)} with quantity > 0 and price > 0.",
        f"Cancelled lines ({status_counts.get('Cancelled', 0)}) are excluded from revenue and units. "
        f"{zeroed} of them were already zeroed out (quantity 0) in the source, so recorded cancelled value understates lost demand.",
        f"Pending lines ({status_counts.get('Pending', 0)}) are not yet confirmed: excluded from sales and reported separately as pipeline value.",
        "'OJ Shipping' lines are treated as accepted orders in transit and counted as sales.",
        "Processed lines with price 0 are excluded as zero-value shipments (likely replacements or samples).",
        "'Carrier Not Mapped' does not affect revenue: those lines count as sales and the carrier is reported as 'Unmapped'.",
        "Masked SKU is the product key (itemName holds internal codes on ~11% of lines).",
    ]
    if partial:
        notes.append(
            f"Partial month(s) {', '.join(p.strftime('%b %Y') for p in partial)} are included in totals but excluded from month-over-month comparison."
        )
    notes.append(
        "Non-moving = SKU appears in the file but sold 0 units (all its lines were cancelled, pending or zero-value). "
        f"Low-moving = sold {LOW_MOVING_MAX_UNITS} unit or fewer in the period. "
        f"Stalled = {STALLED_MIN_UNITS}+ units in the first full month and none in the last."
    )
    notes.append("The 25 vendor SKUs do not appear in the sales sheet; sales analysis runs on the sales sheet as its own dataset.")
    return notes


# --- insights -----------------------------------------------------------------------------


def _facts(r: SalesReport, per_sku: pd.DataFrame, months: list[pd.Period]) -> dict:
    s = r.summary
    sold_skus = per_sku[per_sku["units"] > 0].sort_values("revenue", ascending=False)
    cum = sold_skus["revenue"].cumsum() / sold_skus["revenue"].sum()
    pareto = int((cum <= 0.8).sum()) + 1
    facts = {
        "period": f"{s['period_start']} to {s['period_end']}",
        "net_sales_usd": s["net_sales"],
        "units_sold": s["units_sold"],
        "orders": s["orders"],
        "avg_order_value_usd": s["avg_order_value"],
        "skus_sold": s["skus_sold"],
        "skus_making_80pct_of_revenue": pareto,
        "pct_of_sold_skus_making_80pct_of_revenue": round(pareto / len(sold_skus) * 100, 1),
        "skus_selling_exactly_one_unit": int((sold_skus["units"] == 1).sum()),
        "pct_of_sold_skus_selling_one_unit": round((sold_skus["units"] == 1).mean() * 100, 1),
        "top_sku": {"sku": sold_skus.index[0], "revenue_usd": float(sold_skus["revenue"].iloc[0]),
                    "units": int(sold_skus["units"].iloc[0]), "revenue_share_pct": float(sold_skus["revenue_share_pct"].iloc[0])},
        "top_10_skus_revenue_share_pct": round(float(sold_skus["revenue_share_pct"].head(10).sum()), 1),
        "non_moving_skus": int(len(r.non_moving)),
        "low_moving_skus": int(len(r.low_moving)),
        "stalled_skus": int(len(r.stalled)),
        "cancellation_rate_pct_of_lines": round(s["cancelled_lines"] / s["lines_in_file"] * 100, 1),
        "cancelled_value_recorded_usd": s["cancelled_value_recorded"],
        "pending_value_usd": s["pending_value"],
    }
    full = r.monthly[r.monthly["complete_month"]]
    if len(full) >= 2:
        a, b = full.iloc[-2], full.iloc[-1]
        facts["month_over_month"] = {
            "from": a["month"], "to": b["month"],
            "net_sales_from_usd": float(a["net_sales"]), "net_sales_to_usd": float(b["net_sales"]),
            "net_sales_change_pct": round((b["net_sales"] / a["net_sales"] - 1) * 100, 1),
            "orders_change_pct": round((b["orders"] / a["orders"] - 1) * 100, 1),
        }
    ch = r.by_channel
    facts["top_channels"] = [
        {k: row[k] for k in ("channel", "net_sales", "revenue_share_pct", "cancel_rate_pct", "unmapped_carrier_pct")}
        for _, row in ch.head(5).iterrows()
    ]
    facts["top_3_channels_revenue_share_pct"] = round(float(ch["revenue_share_pct"].head(3).sum()), 1)
    sizeable = ch[ch["lines_total"] >= 100]
    facts["channels_with_most_unmapped_carriers"] = [
        {k: row[k] for k in ("channel", "lines_total", "unmapped_carrier_pct", "cancel_rate_pct")}
        for _, row in sizeable.sort_values("unmapped_carrier_pct", ascending=False).head(3).iterrows()
    ]
    facts["carriers"] = [
        {k: row[k] for k in ("carrier", "revenue_share_pct", "avg_line_value")} for _, row in r.by_carrier.iterrows()
    ]
    facts["fulfilment_routes"] = [
        {k: row[k] for k in ("fulfilment_route", "revenue_share_pct")} for _, row in r.by_route.iterrows()
    ]
    facts["top_suppliers"] = [
        {k: row[k] for k in ("supplier", "revenue_share_pct")} for _, row in r.by_supplier.head(3).iterrows()
    ]
    return facts


INSIGHT_SYSTEM = (
    "You are a senior e-commerce analyst for a US furniture and home-decor wholesaler that sells through "
    "marketplace and retail channels. You write short, specific, decision-oriented insights for the commercial team. "
    "Use ONLY numbers that appear in the provided facts, copied exactly (you may round to fewer decimals or "
    "write $1.39M for 1395310.17). Never compute new numbers, never invent causes as facts; label hypotheses as such."
)


def generate_insights(report: SalesReport, llm: LLM | None) -> None:
    if llm is not None:
        feedback = ""
        for _ in range(2):
            prompt = (
                "Write 4 business insights from these sales facts. Cover different themes "
                "(growth trend, SKU concentration / assortment, channel or operations risk, inventory / non-moving stock). "
                f"{feedback}\n\nFACTS (JSON):\n{_json(report.facts)}"
            )
            try:
                result = llm.structured(system=INSIGHT_SYSTEM, prompt=prompt, schema=InsightSet,
                                        namespace="sales_insights", temperature=0.3)
            except LLMError as exc:
                log.warning("Insight generation failed, using rule-based insights: %s", exc)
                break
            bad = ungrounded_numbers(result.value, report.facts)
            if not bad:
                report.insights = result.value.insights
                report.insights_source = f"{result.provider}:{result.model}" + (" (cached)" if result.cached else "")
                return
            feedback = (f"\nYour previous answer used numbers not present in the facts: {bad}. "
                        "Use only numbers from the facts.")
            log.info("Insights contained ungrounded numbers %s; retrying", bad)
    report.insights = rule_based_insights(report.facts)
    report.insights_source = "rule-based fallback"


def ungrounded_numbers(insights: InsightSet, facts: dict) -> list[str]:
    return [tok for ins in insights.insights for tok in ungrounded(f"{ins.headline} {ins.detail}", facts)]


def rule_based_insights(f: dict) -> list[Insight]:
    out = []
    if mom := f.get("month_over_month"):
        direction = "grew" if mom["net_sales_change_pct"] >= 0 else "fell"
        out.append(Insight(
            headline=f"Net sales {direction} {abs(mom['net_sales_change_pct'])}% month over month",
            detail=f"Net sales went from ${mom['net_sales_from_usd']:,.0f} ({mom['from']}) to ${mom['net_sales_to_usd']:,.0f} "
                   f"({mom['to']}); orders changed {mom['orders_change_pct']}%.",
            recommendation="Check which channels drove the change before planning next month's inventory buys.",
        ))
    out.append(Insight(
        headline="Revenue is spread across a long tail of SKUs",
        detail=f"{f['skus_making_80pct_of_revenue']} SKUs ({f['pct_of_sold_skus_making_80pct_of_revenue']}% of sold SKUs) make 80% of revenue, "
               f"and {f['skus_selling_exactly_one_unit']} SKUs ({f['pct_of_sold_skus_selling_one_unit']}%) sold exactly one unit.",
        recommendation="Protect stock depth on the head SKUs and review single-unit SKUs for markdown or delisting.",
    ))
    if f.get("channels_with_most_unmapped_carriers"):
        c = f["channels_with_most_unmapped_carriers"][0]
        out.append(Insight(
            headline=f"{c['channel']} orders often lack a mapped carrier",
            detail=f"{c['unmapped_carrier_pct']}% of {c['channel']} lines have no carrier mapped, with a {c['cancel_rate_pct']}% cancellation rate.",
            recommendation="Fix carrier mapping in the channel integration so shipments can be tracked and costed.",
        ))
    out.append(Insight(
        headline=f"{f['non_moving_skus']} SKUs were ordered but never sold",
        detail=f"Every line for these SKUs was cancelled, pending or zero-value; overall {f['cancellation_rate_pct_of_lines']}% of lines were cancelled.",
        recommendation="Investigate stock availability and listing accuracy for these SKUs.",
    ))
    return out[:5]


def _json(d: dict) -> str:
    return json.dumps(d, indent=1, default=str)
