(() => {
  "use strict";
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const csrf = () => ($('meta[name="csrf-token"]') || {}).content || "";
  const icon = (name, cls = "") => `<svg class="i ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ---- navigation drawer (mobile) ----
  const app = $("[data-app]");
  $$("[data-nav-toggle]").forEach((b) => b.addEventListener("click", () => app.classList.add("nav-open")));
  $$("[data-nav-close]").forEach((b) => b.addEventListener("click", () => app.classList.remove("nav-open")));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") app && app.classList.remove("nav-open"); });

  // ---- whole-row links (the first-cell link stays the keyboard target) ----
  $$("tr[data-href]").forEach((tr) => tr.addEventListener("click", (e) => {
    if (e.target.closest("a, button, input, select, textarea") || getSelection().toString()) return;
    if (tr.dataset.href.startsWith("#")) { location.hash = tr.dataset.href; return; }
    location.href = tr.dataset.href;
  }));

  // ---- confirm + autosubmit ----
  $$("form[data-confirm]").forEach((f) => f.addEventListener("submit", (e) => { if (!confirm(f.dataset.confirm)) e.preventDefault(); }));
  $$("[data-autosubmit]").forEach((s) => s.addEventListener("change", () => s.form.submit()));

  // ---- new run form ----
  const newRun = $("[data-new-run]");
  if (newRun) {
    const dz = $("[data-dropzone]"), input = $("[data-dz-input]"), name = $("[data-dz-name]"), hint = $("[data-dz-hint]");
    const sources = $$("[data-source]");
    const syncSource = () => {
      const sample = sources.some((r) => r.checked && r.value === "sample");
      dz.style.opacity = sample ? "0.5" : "";
      dz.setAttribute("aria-disabled", sample ? "true" : "false");
    };
    const pick = (file) => {
      if (!file) return;
      name.textContent = file.name;
      const mb = file.size / 1024 / 1024;
      const bad = !/\.xlsx$/i.test(file.name) ? "This is not an .xlsx file." : mb > 25 ? "Larger than the 25 MB limit." : "";
      hint.textContent = bad || `${mb >= 1 ? mb.toFixed(1) + " MB" : Math.ceil(file.size / 1024) + " KB"} · ready to upload`;
      hint.style.color = bad ? "var(--bad)" : "";
      const up = sources.find((r) => r.value === "upload");
      if (up) { up.checked = true; syncSource(); }
    };
    input.addEventListener("change", () => pick(input.files[0]));
    ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("is-over"); }));
    dz.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files[0];
      if (!file) return;
      const dt = new DataTransfer(); dt.items.add(file); input.files = dt.files; pick(file);
    });
    sources.forEach((r) => r.addEventListener("change", syncSource));
    syncSource();
    const skus = $("[data-skus]");
    $$("[data-scope]").forEach((r) => r.addEventListener("change", () => { skus.hidden = !(r.checked && r.value === "selected"); if (!skus.hidden) $("textarea", skus).focus(); }));
    newRun.addEventListener("submit", () => {
      const btn = $("[data-submit]");
      btn.disabled = true;
      btn.innerHTML = `${icon("loader", "spinner")}Starting…`;
    });
  }

  // ---- live run status ----
  const live = $$("[data-run-live]");
  const ACTIVE = ["queued", "running", "cancelling"];
  const BADGES = {
    queued: ["Queued", "badge-live", "clock"], running: ["Running", "badge-live", "loader"],
    cancelling: ["Cancelling", "badge-warn", "loader"], succeeded: ["Completed", "badge-ok", "check"],
    completed_with_errors: ["Completed with errors", "badge-warn", "alert"], failed: ["Failed", "badge-bad", "x"],
    cancelled: ["Cancelled", "", "stop"],
  };
  const STEP_ICON = { ok: "check", failed: "x", running: "loader", skipped: "minus", cancelled: "stop" };
  const secs = (s) => (s == null ? "—" : s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`);

  function stepMeta(st) {
    if (st.status === "running" && st.total) return `${st.done ?? 0} / ${st.total}${st.item ? " · " + st.item : ""}`;
    if (st.status === "running") return "In progress";
    if (st.status === "ok") return secs(st.seconds);
    if (st.status === "failed") return st.detail === "stopped here" ? "Stopped here" : "Failed";
    if (st.status === "cancelled") return st.seconds ? `Cancelled at ${secs(st.seconds)}` : "Cancelled";
    return { skipped: "Skipped", not_started: "Not started" }[st.status] || "Waiting";
  }

  function paint(root, data) {
    const run = data.run;
    $$(".tl-step", root).forEach((el) => {
      const st = run.steps[el.dataset.step];
      if (!st) return;
      if (el.dataset.status !== st.status) {
        el.dataset.status = st.status;
        $("[data-dot]", el).innerHTML = STEP_ICON[st.status] ? icon(STEP_ICON[st.status]) : "";
      }
      $("[data-meta]", el).textContent = stepMeta(st);
      const bar = $(".tl-bar i", el);
      if (bar) bar.style.transform = `scaleX(${st.total ? Math.min(1, (st.done || 0) / st.total) : 0})`;
    });
    const badge = $("[data-run-badge]");
    const b = BADGES[run.status];
    if (badge && b) { badge.className = `badge ${b[1]}`; badge.innerHTML = icon(b[2], b[2] === "loader" ? "spinner" : "") + esc(b[0]); }
    $$("[data-elapsed]").forEach((e) => (e.textContent = data.elapsed));
    const log = $("[data-log]");
    if (log && data.log) {
      const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
      log.innerHTML = data.log.map((l) => `<span class="${l.includes(" WARNING ") ? "w" : l.includes(" ERROR ") ? "e" : ""}">${esc(l)}</span>\n`).join("");
      if (atBottom) log.scrollTop = log.scrollHeight;
    }
    const cancel = $("[data-cancel]");
    if (cancel && run.status === "cancelling") cancel.disabled = true;
  }

  live.forEach((root) => {
    if (!ACTIVE.includes(root.dataset.status)) return;
    const id = root.dataset.runLive;
    let delay = 1500;
    const tick = async () => {
      try {
        const res = await fetch(`/api/runs/${id}/status`, { headers: { Accept: "application/json" }, cache: "no-store" });
        if (res.status === 401) { location.href = `/login?next=${encodeURIComponent(location.pathname + location.search)}`; return; }
        if (!res.ok) throw new Error(res.status);
        const data = await res.json();
        paint(root, data);
        delay = 1500;
        if (!ACTIVE.includes(data.run.status)) { setTimeout(() => location.reload(), 700); return; }
      } catch (err) {
        delay = Math.min(delay * 2, 15000);
      }
      setTimeout(tick, delay);
    };
    setTimeout(tick, 800);
  });

  $$("[data-cancel]").forEach((btn) => btn.addEventListener("click", async () => {
    if (!confirm("Cancel this run? Work finished so far stays cached.")) return;
    btn.disabled = true;
    btn.innerHTML = `${icon("loader", "spinner")}<span class="btn-label">Cancelling…</span>`;
    try {
      const res = await fetch(`/runs/${btn.dataset.cancel}/cancel`, { method: "POST", headers: { "X-CSRF-Token": csrf() } });
      if (!res.ok) throw new Error();
    } catch {
      btn.disabled = false;
      btn.innerHTML = `${icon("stop")}<span class="btn-label">Cancel run</span>`;
      alert("Could not cancel the run. Check your connection and try again.");
    }
  }));

  // ---- settings: test keys ----
  $$("[data-test-key]").forEach((btn) => btn.addEventListener("click", async () => {
    const key = btn.dataset.testKey;
    const out = $(`[data-test-result="${key}"]`);
    btn.disabled = true;
    out.className = "test-result";
    out.innerHTML = `${icon("loader", "spinner")}<span>Testing…</span>`;
    try {
      const res = await fetch(`/api/settings/test/${key}`, { method: "POST", headers: { "X-CSRF-Token": csrf() } });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Test failed");
      out.className = `test-result ${data.ok ? "ok" : "bad"}`;
      out.innerHTML = `${icon(data.ok ? "check" : "alert")}<span>${esc(data.message)}${data.detail ? `<span class="detail">${esc(data.detail)}</span>` : ""}</span>`;
    } catch (err) {
      out.className = "test-result bad";
      out.innerHTML = `${icon("alert")}<span>${esc(err.message || "Could not run the test.")}</span>`;
    } finally {
      btn.disabled = false;
    }
  }));

  // ---- charts: hover/focus tooltips ----
  $$("[data-chart]").forEach((chart) => {
    const tip = $("[data-tipbox]", chart);
    const show = (col) => {
      const [title, ...rest] = col.dataset.tip.split("|");
      tip.innerHTML = `<strong>${esc(title)}</strong><br>${rest.map(esc).join("<br>")}`;
      const bar = $(".col-bar", col).getBoundingClientRect(), box = chart.getBoundingClientRect();
      tip.style.left = `${Math.min(Math.max(bar.left - box.left + bar.width / 2, 90), box.width - 90)}px`;
      tip.style.top = `${bar.top - box.top}px`;
      tip.classList.add("on");
    };
    const hide = () => tip.classList.remove("on");
    $$("[data-tip]", chart).forEach((col) => {
      col.setAttribute("tabindex", "0");
      col.setAttribute("aria-label", col.dataset.tip.replace(/\|/g, ", "));
      col.addEventListener("mouseenter", () => show(col));
      col.addEventListener("focus", () => show(col));
      col.addEventListener("mouseleave", hide);
      col.addEventListener("blur", hide);
    });
  });

  // ---- competitor benchmarks: open the one a summary row points at ----
  const openBench = () => {
    const el = location.hash && document.getElementById(location.hash.slice(1));
    if (el && el.tagName === "DETAILS") { el.open = true; el.scrollIntoView({ block: "start" }); }
  };
  window.addEventListener("hashchange", openBench);
  openBench();
  $$("[data-toggle-benches]").forEach((btn) => btn.addEventListener("click", () => {
    const all = $$("details.bench"), open = !all.every((d) => d.open);
    all.forEach((d) => (d.open = open));
    btn.textContent = open ? "Collapse all" : "Expand all";
  }));

  // ---- sub-tabs ----
  $$("[data-subtabs]").forEach((box) => {
    const tabs = $$("[data-subtab]", box);
    tabs.forEach((t) => t.addEventListener("click", () => {
      tabs.forEach((x) => x.setAttribute("aria-selected", x === t ? "true" : "false"));
      $$("[data-pane]", box).forEach((p) => (p.hidden = p.dataset.pane !== t.dataset.subtab));
    }));
  });

  // ---- product viewer ----
  const viewer = $("[data-viewer]");
  if (viewer) {
    const img = $("[data-view-img]", viewer);
    $$("[data-view]", viewer).forEach((b) => b.addEventListener("click", () => {
      $$("[data-view]", viewer).forEach((x) => x.setAttribute("aria-pressed", x === b ? "true" : "false"));
      img.src = b.dataset.view === "src" ? img.dataset.src : img.dataset.life;
    }));
    document.addEventListener("keydown", (e) => {
      if (/input|textarea|select/i.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
      const link = e.key === "ArrowLeft" ? $("[data-key-prev]") : e.key === "ArrowRight" ? $("[data-key-next]") : null;
      if (link && link.getAttribute("href")) location.href = link.getAttribute("href");
    });
  }
})();
