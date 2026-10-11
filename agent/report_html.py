"""Single-file HTML report (no external assets). Images are referenced from ../images/."""
from __future__ import annotations

import datetime as dt
import os
from html import escape as e
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .graph import PipelineState, RunContext

CSS = """
:root{--ink:#1d2327;--muted:#5b6670;--line:#e3e1dc;--bg:#faf9f6;--card:#fff;--accent:#2f3e46;--ok:#2e7d4f;--warn:#a86b00;--bad:#b23b3b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 80px}
header{border-bottom:2px solid var(--ink);padding-bottom:16px;margin-bottom:28px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:20px;margin:44px 0 14px;padding-top:8px;border-top:1px solid var(--line)}
header+h2{border-top:0;margin-top:8px}
h3{font-size:16px;margin:0 0 6px}.muted{color:var(--muted)}.small{font-size:13px}
nav{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:12px;font-size:14px}nav a{color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}.kpi b{display:block;font-size:20px}
.insight{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:6px;padding:12px 16px;margin:10px 0}
.product{display:grid;grid-template-columns:minmax(0,420px) minmax(0,1fr);gap:20px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}
.product img{width:100%;height:auto;border-radius:6px;display:block;background:#eee}
.imgs{display:grid;gap:8px}.imgs .src{width:96px;border:1px solid var(--line)}
.product ul{padding-left:18px;margin:8px 0}.product li{margin:4px 0}
.chip{display:inline-block;font-size:12px;border-radius:999px;padding:1px 9px;border:1px solid currentColor;margin-right:6px}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
details{margin-top:8px}summary{cursor:pointer;color:var(--accent);font-size:14px}
table{border-collapse:collapse;width:100%;background:var(--card);font-size:13.5px}
th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}th{background:#f1efea;font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
.attrs td:first-child{color:var(--muted);white-space:nowrap;width:1%}
@media (max-width:760px){.product{grid-template-columns:1fr}h1{font-size:23px}}
"""


def render(state: "PipelineState", ctx: "RunContext") -> str:
    report = state.get("sales_report")
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Catalog Agent Report</title><style>{CSS}</style></head><body><main>"]
    parts.append(
        "<header><h1>Catalog Agent Report</h1>"
        f"<div class='muted'>Input <code>{e(ctx.settings.input_path.name)}</code> &middot; generated {dt.datetime.now():%d %b %Y %H:%M}</div>"
        "<nav><a href='#sales'>Sales analysis</a><a href='#content'>Product content</a><a href='#competitors'>Competitors</a>"
        "<a href='#quality'>Data quality</a><a href='#run'>Run log</a></nav></header>"
    )
    parts.append(_sales(report) if report else "<h2 id='sales'>Sales analysis</h2><p class='bad'>Not available (step failed, see run log).</p>")
    images_href = os.path.relpath(ctx.settings.images_dir, ctx.settings.output_dir).replace(os.sep, "/")
    parts.append(_content(state, images_href))
    parts.append(_competitors(state))
    parts.append(_quality(ctx))
    parts.append(_run_log(state))
    parts.append("</main></body></html>")
    return "".join(parts)


def _sales(r) -> str:
    s = r.summary
    kpis = [("Net sales", f"${s['net_sales']:,.0f}"), ("Units sold", f"{s['units_sold']:,}"), ("Orders", f"{s['orders']:,}"),
            ("Avg order value", f"${s['avg_order_value']:,.2f}"), ("SKUs sold", f"{s['skus_sold']:,}"),
            ("Lines excluded", f"{s['lines_excluded']:,}")]
    out = [f"<h2 id='sales'>Sales analysis</h2><p class='muted small'>{e(s['period_start'])} to {e(s['period_end'])}. "
           f"Cancelled and pending orders excluded; see how sales were counted below.</p><div class='kpis'>"]
    out += [f"<div class='kpi'><span class='muted small'>{e(k)}</span><b>{e(v)}</b></div>" for k, v in kpis]
    out.append(f"</div><h3 style='margin-top:22px'>Business insights <span class='muted small'>({e(r.insights_source)})</span></h3>")
    for ins in r.insights:
        out.append(f"<div class='insight'><b>{e(ins.headline)}</b><div>{e(ins.detail)}</div>"
                   f"<div class='small'><b>Action:</b> {e(ins.recommendation)}</div></div>")
    out.append("<h3 style='margin-top:22px'>Top SKUs by revenue</h3>")
    out.append(_table(r.top_by_revenue[["sku", "revenue", "units", "orders", "avg_price", "revenue_share_pct"]],
                      money={"revenue", "avg_price"}))
    out.append("<h3 style='margin-top:22px'>Low and non-moving SKUs</h3>"
               f"<p class='small'>{len(r.non_moving):,} non-moving (ordered, never sold) &middot; {len(r.low_moving):,} low-moving "
               f"(1 unit in the period) &middot; {len(r.stalled):,} stalled (selling in July, nothing in August). Full lists in the Excel file.</p>")
    out.append(_table(r.non_moving.head(10), money={"lost_value_recorded"}))
    out.append("<h3 style='margin-top:22px'>Channels</h3>")
    out.append(_table(r.by_channel.head(10), money={"net_sales"}))
    out.append("<details><summary>How sales were counted</summary><ul>" + "".join(f"<li>{e(n)}</li>" for n in r.treatment_notes) + "</ul></details>")
    return "".join(out)


def _content(state: "PipelineState", images_href: str) -> str:
    content = state.get("content", {})
    images = state.get("images", {})
    prices = state.get("our_prices", {})
    out = ["<h2 id='content'>Product content &amp; lifestyle images</h2>"]
    for p in state.get("products", []):
        c = content.get(p.sku)
        img = images.get(p.sku)
        if img and img.status == "ok":
            img_html = f"<img src='{e(images_href)}/{e(p.sku)}.jpg' alt='Lifestyle image of {e(c.title if c else p.sku)}' loading='lazy'>"
        else:
            img_html = f"<p class='muted small'>No lifestyle image: {e(img.note if img else 'step not run')}</p>"
        src = f"<img class='src' src='{e(p.image_url)}' alt='Vendor source photo' loading='lazy'>" if p.image_url else ""
        status = c.qa_status if c else "not run"
        cls = {"passed": "ok", "passed_with_warnings": "warn"}.get(status, "bad")
        price = prices.get(p.sku)
        price_html = f"${price.price:,.2f} on benzara.com" if price and price.price else "not listed on benzara.com"
        out.append(f"<article class='product'><div class='imgs'>{img_html}<div class='small muted'>Source photo {src}</div>"
                   f"<div class='small muted'>{e(img.provider if img else '')}</div></div><div>")
        out.append(f"<div class='small muted'>{e(p.sku)} &middot; {e(price_html)}</div><h3>{e(c.title if c else p.sku)}</h3>"
                   f"<span class='chip {cls}'>QA {e(status.replace('_', ' '))}</span>"
                   f"<span class='muted small'>{e(c.model if c else '')}</span>")
        if c:
            out.append(f"<p>{e(c.description)}</p><ul>")
            for f in c.key_features:
                label, _, text = f.partition(": ")
                out.append(f"<li><b>{e(label)}:</b> {e(text)}</li>")
            out.append("</ul><details><summary>Attributes</summary><table class='attrs'>"
                       + "".join(f"<tr><td>{e(k)}</td><td>{e(v)}</td></tr>" for k, v in c.attributes.items()) + "</table></details>")
            if c.qa_issues:
                out.append("<p class='small warn'>QA notes: " + e("; ".join(c.qa_issues)) + "</p>")
        out.append("</div></article>")
    return "".join(out)


def _competitors(state: "PipelineState") -> str:
    out = ["<h2 id='competitors'>Competitor benchmarking</h2><p class='muted small'>Our price = live price on benzara.com. "
           "Competitors = real Google Shopping listings (SerpAPI), picked for comparability. One model per collection; "
           "colour variants share the benchmark.</p>"]
    rows = state.get("competitor_rows", [])
    for s in state.get("competitor_summaries", []):
        our = f"${s.our_price.price:,.2f}" if s.our_price.price else "not listed"
        out.append(f"<div class='insight'><b>{e(s.sku)}</b> <span class='muted small'>(covers {e(', '.join(s.covers_skus))})</span>"
                   f"<div class='small'>{e(s.product)}</div><div class='small'>Our price: <b>{e(our)}</b> &middot; {e(s.positioning)}"
                   + (f" &middot; suggested launch band <b>{e(s.suggested_price_band)}</b>" if s.suggested_price_band else "")
                   + f"</div><div>{e(s.insight)}</div></div>")
        mine = [r for r in rows if r.sku == s.sku]
        if mine:
            df = pd.DataFrame([{"Competitor product": r.competitor_product, "Marketplace": r.marketplace,
                                "Price": r.competitor_price, "Ours - theirs": r.price_diff_usd, "Diff %": r.price_diff_pct}
                               for r in mine])
            out.append(_table(df, money={"Price", "Ours - theirs"}, links=[r.link for r in mine]))
    return "".join(out)


def _quality(ctx: "RunContext") -> str:
    df = pd.DataFrame(ctx.quality.to_records())
    if df.empty:
        return "<h2 id='quality'>Data quality</h2><p>No issues found.</p>"
    return ("<h2 id='quality'>Data quality</h2><p class='muted small'>Every anomaly the pipeline found and what it did. "
            "The input file was not edited by hand.</p>" + _table(df[["sheet", "severity", "field", "issue", "count", "action", "examples"]]))


def _run_log(state: "PipelineState") -> str:
    return "<h2 id='run'>Run log</h2>" + _table(pd.DataFrame(state.get("steps", [])))


def _table(df: pd.DataFrame, money: set[str] = frozenset(), links: list[str | None] | None = None) -> str:
    if df.empty:
        return "<p class='muted small'>None.</p>"
    cols = list(df.columns)
    numeric = {c for c in cols if pd.api.types.is_numeric_dtype(df[c])}
    head = "".join(f"<th class='{'num' if c in numeric else ''}'>{e(str(c).replace('_', ' '))}</th>" for c in cols)
    body = []
    for i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v) if not isinstance(v, (list, dict)) else False:
                text = ""
            elif c in money and isinstance(v, (int, float)):
                text = f"${v:,.2f}"
            elif isinstance(v, float):
                text = f"{v:,.2f}"
            elif isinstance(v, pd.Timestamp):
                text = f"{v:%Y-%m-%d}"
            else:
                text = str(v)
            cell = e(text)
            if links and c == cols[0] and links[i]:
                cell = f"<a href='{e(links[i])}' rel='noopener' target='_blank'>{cell}</a>"
            cells.append(f"<td class='{'num' if c in numeric else ''}'>{cell}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
