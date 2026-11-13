// Total-demand chart and the per-series explorer.
import { el, segControl, methodChips, seriesColor, fmtPct, SEGMENTS, SEGMENT_LABEL } from "./format.js";
import { lineChart } from "./charts.js";

const MAX_METHODS = 3;
const DEFAULT_METHODS = ["lgbm_ratio", "chronos_bolt", "seasonal_naive"];

function legend(container, methods, selected, withBand = true) {
  const byId = Object.fromEntries(methods.map((m) => [m.id, m]));
  const items = [
    el("span", {}, [el("i", { style: "--c:var(--s-actual)" }), "Actual"]),
    ...selected.map((id) => el("span", {}, [el("i", { style: `--c:${seriesColor(id)}` }), byId[id].label])),
  ];
  if (withBand) items.push(el("span", {}, [el("i", { class: "band", style: `--c:${seriesColor(selected[0])}` }), `10–90% range, ${byId[selected[0]].label}`]));
  container.replaceChildren(...items);
}

/** history + test arrays -> chart inputs; forecasts are null over the history. */
function chartInput(histWeeks, hist, testWeeks, actual, fc, methods, selected) {
  const byId = Object.fromEntries(methods.map((m) => [m.id, m]));
  const pad = histWeeks.map(() => null);
  const dates = [...histWeeks, ...testWeeks];
  const lines = [
    { id: "actual", label: "Actual", color: "var(--s-actual)", values: [...hist, ...actual], width: 1.6 },
    ...selected.map((id) => ({ id, label: byId[id].label, color: seriesColor(id), values: [...pad, ...fc[id].f] })),
  ];
  const first = fc[selected[0]];
  const band = first.q10.every((v) => v === null) ? null : { lo: [...pad, ...first.q10], hi: [...pad, ...first.q90], color: seriesColor(selected[0]) };
  return { dates, cutIndex: histWeeks.length - 1, lines, band };
}

/** Weekly totals for one segment: actuals + selected methods, no range (percentiles don't sum). */
export function drawTotals(chartEl, legendEl, summary, segment, selected) {
  const methods = summary.methods;
  const t = summary.totals[segment];
  const n = t.weeks.length - summary.meta.horizon;
  legend(legendEl, methods, selected, false);
  const input = chartInput(t.weeks.slice(0, n), t.actual.slice(0, n), t.weeks.slice(n), t.actual.slice(n), t.forecast, methods, selected);
  lineChart(chartEl, { ...input, band: null, zeroBase: false });
}

export function initTotals(summary) {
  const methods = summary.methods;
  let segment = "all";
  let selected = [...DEFAULT_METHODS];
  const draw = () => drawTotals(document.getElementById("total-chart"), document.getElementById("total-legend"), summary, segment, selected);
  segControl(document.getElementById("total-seg"), SEGMENTS.map((s) => [s, SEGMENT_LABEL[s]]), segment, (v) => { segment = v; draw(); });
  methodChips(document.getElementById("total-methods"), methods, selected, MAX_METHODS, (sel) => { selected = sel; draw(); });
  draw();
}

export function initExplorer(explorer, summary, api) {
  const methods = summary.methods;
  let selected = [...DEFAULT_METHODS];
  let segment = "all";
  let current = explorer.series[0];
  const list = document.getElementById("series-list");

  const renderList = () => {
    const items = explorer.series.filter((s) => segment === "all" || s.segment === segment);
    list.replaceChildren(
      ...items.map((s) =>
        el("button", { type: "button", class: "series-item", role: "listitem", "aria-current": String(s.id === current.id), onclick: () => { current = s; renderList(); drawSeries(); } }, [
          el("span", {}, s.id),
          el("span", { class: "pill" }, s.segment),
        ]),
      ),
    );
  };

  const drawSeries = () => {
    const s = current;
    document.getElementById("series-head").replaceChildren(
      el("h2", {}, s.id),
      el("span", { class: "pill" }, s.segment),
      el("span", { class: "muted small" }, `${Math.round(s.actual.reduce((a, b) => a + b, 0))} units in the 12 test weeks`),
    );
    legend(document.getElementById("series-legend"), methods, selected);
    lineChart(document.getElementById("series-chart"), { ...chartInput(s.history_weeks, s.history, s.test_weeks, s.actual, s.forecast, methods, selected), height: 280 });
    const ranked = methods.map((m) => [m, s.forecast[m.id].wape]).filter(([, w]) => w !== null).sort((a, b) => a[1] - b[1]);
    document.getElementById("series-wape").replaceChildren(
      el("span", { class: "eyebrow" }, "Series WAPE"),
      ...ranked.map(([m, w], i) => el("span", {}, i === 0 ? [el("b", {}, m.label), ` ${fmtPct(w, 0)}`] : `${m.label} ${fmtPct(w, 0)}`)),
    );
  };

  segControl(document.getElementById("explorer-seg"), SEGMENTS.map((s) => [s, SEGMENT_LABEL[s]]), segment, (v) => {
    segment = v;
    const pool = explorer.series.filter((s) => v === "all" || s.segment === v);
    if (!pool.includes(current)) current = pool[0];
    renderList();
    drawSeries();
  });
  methodChips(document.getElementById("explorer-methods"), methods, selected, MAX_METHODS, (sel) => { selected = sel; drawSeries(); });
  renderList();
  drawSeries();
  if (api) initSearch(api, (s) => { current = s; renderList(); drawSeries(); });
}

/** Only when the local backend is running: look up any of the ~30k series by id. */
function initSearch(api, onLoad) {
  const wrap = document.getElementById("search-wrap");
  const input = document.getElementById("series-search");
  wrap.hidden = false;
  document.getElementById("explorer-sub").textContent += " The local server is running, so you can also load any series by id.";
  let timer = 0;
  let controller = null;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const id = input.value.trim().toUpperCase();
      if (!/^[A-Z0-9_]{6,40}$/.test(id)) return;
      controller?.abort();
      controller = new AbortController();
      try {
        const res = await fetch(`${api}/series/${encodeURIComponent(id)}`, { signal: controller.signal });
        if (res.status === 404) { input.setCustomValidity("No series with that id"); input.reportValidity(); return; }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        input.setCustomValidity("");
        onLoad(await res.json());
      } catch (err) {
        if (err.name !== "AbortError") { input.setCustomValidity("Lookup failed. Is the local server still running?"); input.reportValidity(); }
      }
    }, 350);
  });
}
