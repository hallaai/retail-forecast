// Score tables: method x segment heat table, stability across windows, level vs ratio target.
import { el, fmtPct, fmtSigned, fmtNum, fmtInt, SEGMENTS, SEGMENT_LABEL, seriesColor } from "./format.js";

export const METRICS = {
  wape: { label: "WAPE", fmt: (v) => fmtPct(v), better: "low", note: "Σ|actual − forecast| / Σ actual over every series-week in the segment. Volume-weighted." },
  mase: { label: "MASE", fmt: (v) => fmtNum(v), better: "low", note: "Mean over series of test MAE ÷ in-sample naive MAE. Every series counts equally; 1.0 = as good as last-week naive in-sample." },
  bias: { label: "Bias", fmt: (v) => fmtSigned(v), better: "zero", note: "Σ(forecast − actual) / Σ actual. Negative = under-forecast (lost sales risk), positive = over-forecast (excess stock)." },
  swql: { label: "Pinball", fmt: (v) => fmtPct(v), better: "low", note: "Mean pinball loss at the 10th, 50th and 90th percentiles, scaled by Σ actual and ×2 so the median term alone equals WAPE. Scores the whole range, not just the point." },
  coverage80: { label: "80% coverage", fmt: (v) => fmtPct(v, 0), better: "target", target: 0.8, note: "Share of actuals inside the 10th–90th percentile range. Target 80%." },
};

/** Look up scores by method/window/segment; "bt" averages the backtest windows. */
export function scoreIndex(scores) {
  const idx = new Map(scores.map((r) => [`${r.method}|${r.window}|${r.segment}`, r]));
  const windows = [...new Set(scores.map((r) => r.window))].filter((w) => w !== "test");
  return (method, window, segment, metric) => {
    if (window === "bt") {
      const vals = windows.map((w) => idx.get(`${method}|${w}|${segment}`)?.[metric]).filter((v) => v !== null && v !== undefined);
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    }
    return idx.get(`${method}|${window}|${segment}`)?.[metric] ?? null;
  };
}

function goodness(metric, value, column) {
  const spec = METRICS[metric];
  const vals = column.filter((v) => v !== null);
  if (value === null || !vals.length) return null;
  if (spec.better === "low") {
    const best = Math.min(...vals), worst = Math.max(...vals);
    return worst === best ? 1 : 1 - (value - best) / (worst - best);
  }
  const dist = (v) => Math.abs(v - (spec.target ?? 0));
  const best = Math.min(...vals.map(dist)), worst = Math.max(...vals.map(dist));
  return worst === best ? 1 : 1 - (dist(value) - best) / (worst - best);
}

function cellStyle(metric, value, g) {
  if (g === null) return "";
  if (metric === "bias") {
    const rgb = value > 0 ? "var(--heat-over)" : "var(--heat)";
    const a = Math.min(1, Math.abs(value) * 2.6);
    return `background: rgba(${rgb}, calc(var(--heat-max) * ${a.toFixed(3)}))`;
  }
  return `background: rgba(var(--heat), calc(var(--heat-max) * ${(0.1 + 0.9 * g).toFixed(3)}))`;
}

const methodCell = (m, extra = null) =>
  el("td", {}, [el("span", { class: "method-name" }, [el("span", { class: "dot", style: `--c:${seriesColor(m.id)}` }), m.label]), extra]);

export function renderScoreTable(table, { methods, get, metric, window, meta }) {
  const segInfo = meta.segments;
  const head = el("tr", {}, [
    el("th", { scope: "col" }, "Method"),
    ...SEGMENTS.map((s) =>
      el("th", { scope: "col" }, [
        el("span", { class: "seg-name" }, SEGMENT_LABEL[s]),
        el("span", { class: "sub" }, s === "all" ? `${fmtInt(meta.n_series)} series` : `${fmtInt(segInfo[s].series)} · ${fmtPct(segInfo[s].volume_share, 0)} of units`),
      ]),
    ),
  ]);
  const columns = Object.fromEntries(SEGMENTS.map((s) => [s, methods.map((m) => get(m.id, window, s, metric))]));
  const rows = methods.map((m, i) => {
    const cells = SEGMENTS.map((s) => {
      const v = columns[s][i];
      const g = goodness(metric, v, columns[s]);
      return el("td", { class: g === 1 ? "best" : "", style: cellStyle(metric, v, g) }, METRICS[metric].fmt(v));
    });
    const brk = i > 0 && methods[i - 1].family !== m.family;
    const calTag = m.calibrated && (metric === "swql" || metric === "coverage80")
      ? el("span", { class: "family-tag", title: "Range calibrated on backtest residuals" }, "cal.")
      : null;
    return el("tr", { class: brk ? "family-break" : "" }, [methodCell(m, calTag), ...cells]);
  });
  table.replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
}

/**
 * Small comparison table: rows = methods, cols = [{metric, segment, label, sub}].
 * Each cell shows the test value and, in brackets, the backtest mean. ● = best on test in the column.
 */
export function renderCompare(table, { get, rows, cols }) {
  const head = el("tr", {}, [
    el("th", { scope: "col" }, "Variant"),
    ...cols.map((c) => el("th", { scope: "col" }, [el("span", { class: "seg-name" }, c.label), el("span", { class: "sub" }, c.sub)])),
  ]);
  const values = cols.map((c) => rows.map((r) => get(r.id, "test", c.segment, c.metric)));
  const body = rows.map((r, i) =>
    el("tr", {}, [
      el("td", {}, r.label),
      ...cols.map((c, k) => {
        const v = values[k][i];
        const bt = get(r.id, "bt", c.segment, c.metric);
        const g = goodness(c.metric, v, values[k]);
        return el("td", { class: g === 1 ? "best" : "" }, [METRICS[c.metric].fmt(v), el("span", { class: "muted" }, ` (${METRICS[c.metric].fmt(bt)})`)]);
      }),
    ]),
  );
  table.replaceChildren(el("thead", {}, head), el("tbody", {}, body));
}

export function renderStability(table, { methods, get, windows, segment = "all" }) {
  const head = el("tr", {}, [
    el("th", { scope: "col" }, "Method"),
    ...windows.map((w) => el("th", { scope: "col" }, [el("span", { class: "seg-name" }, w.name === "test" ? "Test" : w.name.toUpperCase()), el("span", { class: "sub" }, `${w.from} → ${w.to.slice(5)}`)])),
    el("th", { scope: "col" }, [el("span", { class: "seg-name" }, "Spread"), el("span", { class: "sub" }, "max − min")]),
    el("th", { scope: "col" }, [el("span", { class: "seg-name" }, "Wins"), el("span", { class: "sub" }, `of ${windows.length} windows`)]),
  ]);
  const byWindow = windows.map((w) => methods.map((m) => get(m.id, w.name, segment, "wape")));
  const rows = methods.map((m, i) => {
    const vals = byWindow.map((col) => col[i]);
    const wins = byWindow.filter((col) => Math.min(...col.filter((v) => v !== null)) === col[i]).length;
    const cells = vals.map((v, k) => {
      const g = goodness("wape", v, byWindow[k]);
      return el("td", { class: g === 1 ? "best" : "", style: cellStyle("wape", v, g) }, fmtPct(v));
    });
    return el("tr", {}, [
      methodCell(m),
      ...cells,
      el("td", {}, fmtPct(Math.max(...vals) - Math.min(...vals))),
      el("td", {}, String(wins)),
    ]);
  });
  table.replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
}

export function renderRatioTable(table, { get }) {
  const metrics = ["wape", "mase", "bias"];
  const head = el("tr", {}, [
    el("th", { scope: "col" }, "Segment"),
    ...metrics.flatMap((k) => [
      el("th", { scope: "col" }, [el("span", { class: "seg-name" }, METRICS[k].label), el("span", { class: "sub" }, "level")]),
      el("th", { scope: "col" }, [el("span", { class: "seg-name" }, " "), el("span", { class: "sub" }, "ratio")]),
    ]),
  ]);
  const rows = SEGMENTS.map((s) =>
    el("tr", {}, [
      el("td", {}, SEGMENT_LABEL[s]),
      ...metrics.flatMap((k) => {
        const a = get("lgbm_level", "test", s, k), b = get("lgbm_ratio", "test", s, k);
        const better = k === "bias" ? Math.abs(b) < Math.abs(a) : b < a;
        return [
          el("td", { class: better ? "" : "best" }, METRICS[k].fmt(a)),
          el("td", { class: better ? "best" : "" }, METRICS[k].fmt(b)),
        ];
      }),
    ]),
  );
  table.replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
}
