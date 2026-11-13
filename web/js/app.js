// Entry point: load the pipeline's JSON, render every tab once, route on #hash, toggle theme.
import { el, fmtInt, fmtPct, fmtSigned, fmtDate, segControl, seriesColor, SEGMENTS, SEGMENT_LABEL } from "./format.js";
import { METRICS, scoreIndex, renderScoreTable, renderStability, renderRatioTable, renderCompare } from "./tables.js";
import { costChart, coverageChart } from "./charts.js";
import { renderWapeExample } from "./wape.js";
import { drawTotals, initTotals, initExplorer } from "./explorer.js";
import { renderData } from "./data.js";

const WAPE_BY_SEGMENT = SEGMENTS.map((s) => ({ metric: "wape", segment: s, label: `WAPE ${SEGMENT_LABEL[s].toLowerCase()}`, sub: "test (backtest)" }));

const PAGES = ["overview", "results", "metrics", "tradeoffs", "explorer", "data", "method"];
const status = document.getElementById("load-status");

/* ---------- theme (dark first; the viewer's explicit choice is remembered) ---------- */
function initTheme() {
  const btn = document.getElementById("theme");
  const root = document.documentElement;
  const current = () => root.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const sync = () => { btn.textContent = current() === "dark" ? "Light" : "Dark"; };
  try { const saved = localStorage.getItem("m5-theme"); if (saved) root.dataset.theme = saved; } catch { /* storage blocked */ }
  btn.addEventListener("click", () => {
    root.dataset.theme = current() === "dark" ? "light" : "dark";
    try { localStorage.setItem("m5-theme", root.dataset.theme); } catch { /* storage blocked */ }
    sync();
  });
  sync();
}

/* ---------- routing: plain #name hashes ---------- */
function route() {
  const name = PAGES.includes(location.hash.slice(1)) ? location.hash.slice(1) : "overview";
  document.querySelectorAll("[data-page]").forEach((p) => { p.hidden = p.dataset.page !== name; });
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${name}`));
  window.scrollTo(0, 0);
}

/* ---------- data ---------- */
async function getJson(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

async function detectApi() {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 1500);
  try {
    const res = await fetch("api/health", { signal: ctrl.signal });
    return res.ok ? "api" : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function boot() {
  document.querySelectorAll("[data-page]").forEach((p) => { p.hidden = true; });
  status.hidden = false;
  status.className = "state";
  status.replaceChildren(el("span", {}, "Loading results…"));
  try {
    const [summary, explorer, study, api, profile] = await Promise.all([
      getJson("data/summary.json"), getJson("data/explorer.json"), getJson("data/study.json"), detectApi(),
      getJson("data/profile.json").catch(() => null), // optional: the Data tab says how to create it
    ]);
    status.hidden = true;
    render(summary, explorer, study, api);
    renderData(profile);
    route();
  } catch {
    status.className = "state error";
    status.replaceChildren(
      el("span", {}, "The results could not be loaded. Run scripts/run_pipeline.py first, then reload."),
      el("button", { type: "button", onclick: boot }, "Try again"),
    );
  }
}

/* ---------- pages ---------- */
function render(summary, explorer, study, api) {
  const get = scoreIndex(summary.scores);
  renderOverview(summary, study, get);
  renderResults(summary, get);
  renderMetrics(summary, get);
  renderTradeoffs(summary, study, get);
  initTotals(summary);
  initExplorer(explorer, summary, api);
  renderMethod(study);
  document.getElementById("generated").textContent = `Results generated ${summary.generated}.`;
}

function bestMethod(summary, get) {
  return summary.methods.reduce((best, m) => (get(m.id, "test", "all", "wape") < get(best.id, "test", "all", "wape") ? m : best));
}

function renderOverview(summary, study, get) {
  const { meta } = summary;
  const test = meta.windows.find((w) => w.name === "test");
  const best = bestMethod(summary, get);
  const base = get("seasonal_naive", "test", "all", "wape");
  const bestW = get(best.id, "test", "all", "wape");
  const tiles = [
    [fmtInt(meta.n_series), "item × store series", `${meta.n_weeks} weeks, ${meta.first_week.slice(0, 4)}–${meta.last_week.slice(0, 4)}`],
    [`${meta.horizon} weeks`, "forecast horizon", `${meta.windows.length - 1} backtest windows + test`],
    [`Spring ${test.to.slice(0, 4)}`, "test window", `${fmtDate(test.from)} – ${fmtDate(test.to)}`],
    [fmtPct(bestW), `best test WAPE`, best.label],
    [
      `${fmtInt(((base - bestW) * get(best.id, "test", "all", "volume")) / meta.horizon / 1000)}K`,
      "fewer mis-forecast units per week",
      `vs seasonal naive (${fmtPct(1 - bestW / base, 0)} less error)`,
    ],
    [fmtSigned(get(best.id, "test", "all", "bias")), "bias of the best method", "negative = under-forecast"],
  ];
  document.getElementById("tiles").replaceChildren(
    ...tiles.map(([v, k, d]) => el("div", { class: "card tile" }, [el("div", { class: "v" }, v), el("div", { class: "k" }, k), el("div", { class: "d" }, d)])),
  );
  document.getElementById("findings").replaceChildren(
    ...study.findings.map((f) => el("article", { class: "card finding" }, [el("h2", {}, f.title), el("p", {}, f.body)])),
  );
  renderScoreTable(document.getElementById("overview-table"), { methods: summary.methods, get, metric: "wape", window: "test", meta });
  const selected = [best.id, "seasonal_naive"].filter((v, i, a) => a.indexOf(v) === i);
  drawTotals(document.getElementById("overview-chart"), document.getElementById("overview-legend"), summary, "all", selected);
}

function renderResults(summary, get) {
  const { meta } = summary;
  const methods = summary.methods;
  const test = meta.windows.find((w) => w.name === "test");
  let metric = "wape";
  let windowName = "test";
  const select = document.getElementById("window-select");
  select.replaceChildren(
    el("option", { value: "test" }, `Test (${test.from} → ${test.to})`),
    el("option", { value: "bt" }, "Backtest mean (4 windows)"),
    ...meta.windows.filter((w) => w.name !== "test").map((w) => el("option", { value: w.name }, `${w.name.toUpperCase()} (${w.from} → ${w.to})`)),
  );
  const draw = () => {
    renderScoreTable(document.getElementById("score-table"), { methods, get, metric, window: windowName, meta });
    const extra = metric === "swql" || metric === "coverage80"
      ? " Ranges: Naive, Seasonal naive and AutoETS use native Gaussian intervals; Chronos its own quantiles; “cal.” methods are calibrated on backtest residuals."
      : "";
    document.getElementById("table-note").textContent = METRICS[metric].note + extra;
  };
  segControl(document.getElementById("metric-control"), Object.entries(METRICS).map(([k, v]) => [k, v.label]), metric, (v) => { metric = v; draw(); });
  select.addEventListener("change", () => { windowName = select.value; draw(); });
  draw();
  const drawStability = (segment) => renderStability(document.getElementById("stability-table"), { methods, get, windows: meta.windows, segment });
  segControl(document.getElementById("stability-seg"), SEGMENTS.map((s) => [s, SEGMENT_LABEL[s]]), "all", drawStability);
  drawStability("all");
  renderRatioTable(document.getElementById("ratio-table"), { get });
}

const extraLabel = (summary, id) =>
  (summary.extra_methods || []).find((m) => m.id === id)?.label ?? summary.methods.find((m) => m.id === id)?.label ?? id;

function renderMetrics(summary, get) {
  renderWapeExample(document.getElementById("wape-example"), document.getElementById("wape-out"));
  renderCompare(document.getElementById("median-table"), {
    get,
    rows: [
      { id: "chronos_bolt_median", label: "Median as point" },
      { id: "chronos_bolt", label: "Mean of deciles as point" },
    ],
    cols: [
      { metric: "wape", segment: "all", label: "WAPE", sub: "all series" },
      { metric: "bias", segment: "all", label: "Bias", sub: "all series" },
      { metric: "wape", segment: "sparse", label: "WAPE", sub: "sparse" },
      { metric: "bias", segment: "sparse", label: "Bias", sub: "sparse" },
      { metric: "mase", segment: "all", label: "MASE", sub: "all series" },
    ],
  });
  const rows = [
    ["WAPE", "Σ|y − ŷ| / Σy", "Units mis-forecast per unit sold", "Volume-weighted; hits dominate"],
    ["Bias", "Σ(ŷ − y) / Σy", "Systematic over (+) or under (−) forecasting", "Errors cancel; read with WAPE"],
    ["MASE", "mean over series of MAE ÷ naive MAE", "Typical series, every item counts equally", "Unstable when the naive error is tiny"],
    ["Pinball", "mean over q of Σ max(q·e, (q−1)·e) / Σy", "Quality of the whole range (10th, 50th, 90th pct)", "Needs quantiles, not just a point"],
    ["80% coverage", "share of y inside [q10, q90]", "Whether ranges are honest; target 80%", "Says nothing about range width"],
  ];
  document.getElementById("metric-defs").replaceChildren(
    el("thead", {}, el("tr", {}, ["Metric", "Formula", "Answers", "Watch out"].map((h) => el("th", { scope: "col", style: "text-align:left" }, h)))),
    el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c, i) => el("td", { style: `text-align:left;white-space:${i === 1 ? "nowrap" : "normal"}`, class: i === 1 ? "mono" : "" }, c))))),
  );
}

function renderTradeoffs(summary, study, get) {
  const median = (vals) => {
    const v = [...vals].sort((a, b) => a - b);
    if (!v.length) return null;
    const mid = Math.floor(v.length / 2);
    return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
  };
  // median over windows: robust to a window timed while the machine was busy
  const runs = (id) => Object.values(summary.runtimes[id] || {});
  const runtime = (id) => median(runs(id));
  const parts = summary.ensemble_parts || [];
  costChart(
    document.getElementById("cost-chart"),
    summary.methods
      .map((m) => {
        const ids = m.id === "ensemble" ? parts : [m.id];
        const all = ids.map(runs);
        return {
          label: m.label,
          x: ids.reduce((a, p) => a + (runtime(p) || 0), 0),
          lo: all.reduce((a, r) => a + Math.min(...r), 0),
          hi: all.reduce((a, r) => a + Math.max(...r), 0),
          y: get(m.id, "test", "all", "wape"),
          color: seriesColor(m.id),
        };
      })
      .filter((p) => p.x && p.y !== null),
  );
  renderCompare(document.getElementById("ablation-table"), {
    get,
    rows: ["lgbm_level", "lgbm_level_no_exog", "lgbm_level_price_frozen"].map((id) => ({
      id, label: id === "lgbm_level" ? "All features" : extraLabel(summary, id).replace("LightGBM level, ", ""),
    })),
    cols: [...WAPE_BY_SEGMENT, { metric: "bias", segment: "all", label: "Bias", sub: "test (backtest)" }],
  });
  coverageChart(
    document.getElementById("coverage-chart"),
    summary.methods.map((m) => ({ label: m.label, value: get(m.id, "test", "all", "coverage80"), color: seriesColor(m.id) })).filter((r) => r.value !== null),
  );
  document.getElementById("tradeoff-cards").replaceChildren(
    ...study.tradeoffs.map((t) =>
      el("article", { class: "card" }, [el("h2", {}, t.title), el("dl", { class: "kv" }, t.rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)]))]),
    ),
  );
}

function renderMethod(study) {
  const block = (title, items) => [el("h2", {}, title), el("ul", {}, items.map((i) => el("li", {}, i)))];
  document.getElementById("setup").replaceChildren(...block("How it was run", study.setup));
  document.getElementById("limits").replaceChildren(...block("Limitations", study.limitations));
}

initTheme();
addEventListener("hashchange", route);
boot();
