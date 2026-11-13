// Hand-rolled SVG charts: a time-series line chart with a forecast range, a log-x dot plot,
// and a horizontal dot strip. Colors come from CSS tokens so both themes work.
import { el, svgEl, fmtMonth, fmtDate, fmtInt, fmtNum } from "./format.js";

function niceTicks(max, count = 4) {
  if (!(max > 0)) return [0, 1];
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let v = 0; v <= max + step * 0.001; v += step) ticks.push(+v.toFixed(10));
  if (ticks[ticks.length - 1] < max) ticks.push(ticks[ticks.length - 1] + step);
  return ticks;
}

const tickLabel = (v) => (v >= 10000 ? `${+(v / 1000).toFixed(1)}K` : fmtInt(v) === "0" && v > 0 ? v.toFixed(1) : fmtInt(v));

function attachResize(container, draw) {
  let lastW = 0;
  const ro = new ResizeObserver(() => {
    const w = Math.round(container.clientWidth);
    if (w && w !== lastW) { lastW = w; draw(w); }
  });
  ro.observe(container);
  return ro;
}

/**
 * opts: { dates: [iso], cutIndex, lines: [{id,label,color,values,width?}], band?: {lo,hi,color}, height? }
 * values may contain nulls (e.g. a forecast only exists after the cut).
 */
export function lineChart(container, opts) {
  container.replaceChildren();
  container._ro?.disconnect();
  container._ro = attachResize(container, (width) => drawLine(container, width, opts));
}

function drawLine(container, width, { dates, cutIndex, lines, band, height = 300, zeroBase = true }) {
  container.replaceChildren();
  const m = { t: 12, r: 14, b: 26, l: 44 };
  const W = width, H = height, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const n = dates.length;
  const vals = lines.flatMap((l) => l.values).concat(band ? band.hi : []).filter((v) => v !== null && Number.isFinite(v));
  const lo = zeroBase ? 0 : Math.min(...vals);
  const span = niceTicks(Math.max(1, Math.max(...vals) - lo), 4);
  const step = span[1] - span[0];
  const yMin = zeroBase ? 0 : Math.floor(lo / step) * step;
  const ticks = niceTicks(Math.max(1, Math.max(...vals) - yMin), 4).map((t) => t + yMin);
  const yMax = ticks[ticks.length - 1];
  const x = (i) => m.l + (n <= 1 ? 0 : (i / (n - 1)) * iw);
  const y = (v) => m.t + ih - ((v - yMin) / (yMax - yMin)) * ih;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Weekly units, history and forecast" });

  for (const t of ticks) {
    svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: "var(--grid)", "stroke-width": 1 }));
    const tx = svgEl("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end" });
    tx.textContent = tickLabel(t);
    svg.append(tx);
  }
  // month ticks: first week of a month, thinned to fit
  // a week is a month tick when its month differs from the previous week's; the first week only if it opens its month
  const monthIdx = dates
    .map((d, i) => [d, i])
    .filter(([d], i) => (i === 0 ? Number(d.slice(8, 10)) <= 7 : d.slice(5, 7) !== dates[i - 1].slice(5, 7)))
    .map(([, i]) => i);
  const every = Math.max(1, Math.ceil(monthIdx.length / Math.max(2, Math.floor(iw / 70))));
  monthIdx.filter((_, k) => k % every === 0).forEach((i) => {
    const tx = svgEl("text", { x: x(i), y: H - 6, "text-anchor": "middle" });
    tx.textContent = fmtMonth(dates[i]);
    svg.append(tx);
  });

  if (cutIndex !== undefined && cutIndex !== null) {
    const cx = (x(cutIndex) + x(Math.min(cutIndex + 1, n - 1))) / 2;
    svg.append(svgEl("rect", { x: cx, y: m.t, width: Math.max(0, W - m.r - cx), height: ih, fill: "var(--surface-2)", opacity: 0.55 }));
    const lab = svgEl("text", { x: cx + 6, y: m.t + 12, class: "lbl" });
    lab.textContent = "test window";
    svg.append(lab);
  }

  if (band) {
    const idx = band.lo.map((v, i) => (v === null || band.hi[i] === null ? null : i)).filter((i) => i !== null);
    if (idx.length) {
      const top = idx.map((i) => `${x(i)},${y(band.hi[i])}`);
      const bottom = idx.slice().reverse().map((i) => `${x(i)},${y(band.lo[i])}`);
      svg.append(svgEl("polygon", { points: [...top, ...bottom].join(" "), fill: band.color, opacity: 0.16 }));
    }
  }

  for (const line of lines) {
    const segs = [];
    let cur = [];
    line.values.forEach((v, i) => {
      if (v === null || !Number.isFinite(v)) { if (cur.length) segs.push(cur); cur = []; }
      else cur.push(`${x(i)},${y(v)}`);
    });
    if (cur.length) segs.push(cur);
    for (const s of segs) {
      svg.append(svgEl("polyline", { points: s.join(" "), fill: "none", stroke: line.color, "stroke-width": line.width ?? 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    }
  }

  // hover layer: crosshair + tooltip
  const cross = svgEl("line", { y1: m.t, y2: m.t + ih, stroke: "var(--muted)", "stroke-width": 1, visibility: "hidden" });
  const dots = lines.map((l) => svgEl("circle", { r: 4, fill: l.color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }));
  const hit = svgEl("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
  svg.append(cross, ...dots, hit);
  const tip = el("div", { class: "tooltip", hidden: true });
  container.append(svg, tip);

  const hide = () => { cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); tip.hidden = true; };
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / iw) * (n - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    const rows = [];
    lines.forEach((l, k) => {
      const v = l.values[i];
      if (v === null || !Number.isFinite(v)) { dots[k].setAttribute("visibility", "hidden"); return; }
      dots[k].setAttribute("cx", x(i)); dots[k].setAttribute("cy", y(v)); dots[k].setAttribute("visibility", "visible");
      rows.push(el("div", { class: "tt-row" }, [el("span", {}, [el("i", { style: `--c:${l.color}` }), l.label]), el("b", {}, fmtNum(v, v < 100 ? 1 : 0))]));
    });
    if (band && band.lo[i] !== null && band.lo[i] !== undefined) {
      rows.push(el("div", { class: "tt-row" }, [el("span", {}, "10–90% range"), el("b", {}, `${fmtNum(band.lo[i], 1)} – ${fmtNum(band.hi[i], 1)}`)]));
    }
    tip.replaceChildren(el("div", { class: "tt-date" }, `Week of ${fmtDate(dates[i])}`), ...rows);
    tip.hidden = false;
    const scale = rect.width / W;
    const left = x(i) * scale;
    const tw = tip.offsetWidth;
    tip.style.left = `${left + 12 + tw > rect.width ? left - tw - 12 : left + 12}px`;
    tip.style.top = `${m.t * scale}px`;
  });
}

/** Ranked dot strip: one row per method (sorted by WAPE), dot at runtime on a log axis, WAPE at the right.
 * points: [{label, x: seconds, y: wape, color}] */
export function costChart(container, points) {
  container.replaceChildren();
  container._ro?.disconnect();
  container._ro = attachResize(container, (width) => drawCost(container, width, points));
}

function drawCost(container, W, points) {
  container.replaceChildren();
  const rows = [...points].sort((a, b) => a.y - b.y);
  const rowH = 26, m = { t: 22, r: 62, b: 26, l: 150 };
  const H = m.t + m.b + rows.length * rowH, iw = W - m.l - m.r;
  // log axis spanning the nice ticks (seconds) around the data
  const NICE = [1, 3, 10, 30, 60, 180, 600, 1800, 3600];
  const lo = Math.min(...rows.map((p) => p.x)), hi = Math.max(...rows.map((p) => p.x));
  const tickVals = NICE.filter((v, i) => v <= hi * 1.0001 || NICE[i - 1] < hi).filter((v, i, a) => v >= lo || a[i + 1] > lo);
  const l0 = Math.log10(tickVals[0]), l1 = Math.log10(tickVals[tickVals.length - 1]);
  const x = (v) => m.l + ((Math.log10(Math.max(v, 1)) - l0) / (l1 - l0 || 1)) * iw;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Runtime per method, sorted by test WAPE" });
  const head = svgEl("text", { x: W - 4, y: 12, "text-anchor": "end" });
  head.textContent = "test WAPE";
  svg.append(head);
  for (const v of tickVals) {
    svg.append(svgEl("line", { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b, stroke: "var(--grid)" }));
    const t = svgEl("text", { x: x(v), y: H - 8, "text-anchor": "middle" });
    t.textContent = v >= 60 ? `${v / 60} min` : `${v} s`;
    svg.append(t);
  }
  rows.forEach((p, i) => {
    const cy = m.t + i * rowH + rowH / 2;
    const name = svgEl("text", { x: m.l - 10, y: cy + 4, "text-anchor": "end", class: "lbl" });
    name.textContent = p.label;
    svg.append(name);
    svg.append(svgEl("line", { x1: m.l, x2: x(p.x), y1: cy, y2: cy, stroke: p.color, "stroke-width": 2, opacity: 0.35 }));
    const dot = svgEl("circle", { cx: x(p.x), cy, r: 5, fill: p.color, stroke: "var(--surface)", "stroke-width": 2 });
    const title = svgEl("title");
    const range = p.lo !== undefined ? ` (range ${Math.round(p.lo)}–${Math.round(p.hi)} s)` : "";
    title.textContent = `${p.label}: median ${p.x.toFixed(0)} s per run${range}, test WAPE ${(p.y * 100).toFixed(1)}%`;
    dot.append(title);
    svg.append(dot);
    const val = svgEl("text", { x: W - 4, y: cy + 4, "text-anchor": "end", class: "lbl" });
    val.textContent = `${(p.y * 100).toFixed(1)}%`;
    svg.append(val);
  });
  container.append(svg);
}

/** Horizontal dot strip for coverage: rows [{label, value, color}], target line at `target`. */
export function coverageChart(container, rows, target = 0.8) {
  container.replaceChildren();
  container._ro?.disconnect();
  container._ro = attachResize(container, (W) => {
    container.replaceChildren();
    const rowH = 26, m = { t: 8, r: 50, b: 26, l: 150 };
    const H = m.t + m.b + rows.length * rowH, iw = W - m.l - m.r;
    const lo = 0.6, hi = 1.0;
    const x = (v) => m.l + ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * iw;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Coverage of 80% ranges per method" });
    for (const v of [0.6, 0.7, 0.8, 0.9, 1.0]) {
      svg.append(svgEl("line", { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b, stroke: v === target ? "var(--text-2)" : "var(--grid)", "stroke-width": v === target ? 1.5 : 1 }));
      const t = svgEl("text", { x: x(v), y: H - 8, "text-anchor": "middle" });
      t.textContent = `${v * 100}%`;
      svg.append(t);
    }
    rows.forEach((r, i) => {
      const cy = m.t + i * rowH + rowH / 2;
      const name = svgEl("text", { x: m.l - 10, y: cy + 4, "text-anchor": "end", class: "lbl" });
      name.textContent = r.label;
      svg.append(name);
      svg.append(svgEl("line", { x1: x(target), x2: x(r.value), y1: cy, y2: cy, stroke: r.color, "stroke-width": 2, opacity: 0.5 }));
      svg.append(svgEl("circle", { cx: x(r.value), cy, r: 5, fill: r.color, stroke: "var(--surface)", "stroke-width": 2 }));
      const val = svgEl("text", { x: W - m.r + 8, y: cy + 4, class: "lbl" });
      val.textContent = `${(r.value * 100).toFixed(0)}%`;
      svg.append(val);
    });
    container.append(svg);
  });
}
