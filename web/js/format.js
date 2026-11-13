// Small formatting and DOM helpers shared by every section.

export const SEGMENTS = ["all", "hit", "middle", "sparse", "new"];
export const SEGMENT_LABEL = { all: "All series", hit: "Hits", middle: "Middle", sparse: "Sparse", new: "New" };

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "style") node.setAttribute("style", v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of [].concat(children)) {
    if (c === null || c === undefined) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export const svgEl = (tag, attrs = {}) => {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) node.setAttribute(k, v);
  return node;
};

export const fmtPct = (v, digits = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(digits)}%`);
export const fmtSigned = (v, digits = 1) => (v === null || v === undefined ? "–" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(digits)}%`);
export const fmtNum = (v, digits = 2) => (v === null || v === undefined ? "–" : v.toFixed(digits));
export const fmtInt = (v) => (v === null || v === undefined ? "–" : Math.round(v).toLocaleString("en-US"));
export const fmtSeconds = (s) => (s >= 90 ? `${(s / 60).toFixed(1)} min` : `${s.toFixed(s < 10 ? 1 : 0)} s`);
export const fmtDate = (iso) => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
export const fmtMonth = (iso) => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", { month: "short", year: "2-digit", timeZone: "UTC" });

export const seriesColor = (method) => `var(--s-${method})`;

/** Segmented control. Calls onChange(value) and keeps aria-pressed in sync. */
export function segControl(container, options, value, onChange) {
  container.replaceChildren();
  const buttons = options.map(([val, label]) =>
    el("button", { type: "button", "aria-pressed": String(val === value), onclick: () => select(val) }, label),
  );
  function select(val) {
    buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i][0] === val)));
    onChange(val);
  }
  container.append(...buttons);
}

/** Method toggle chips with a cap on how many can be on at once (oldest selection drops off). */
export function methodChips(container, methods, selected, max, onChange) {
  let current = [...selected];
  container.replaceChildren();
  const chips = methods.map((m) =>
    el("button", { type: "button", class: "chip", "aria-pressed": String(current.includes(m.id)), style: `--c:${seriesColor(m.id)}`, onclick: () => toggle(m.id) }, [
      el("span", { class: "sw" }),
      m.label,
    ]),
  );
  function toggle(id) {
    if (current.includes(id)) {
      if (current.length === 1) return;
      current = current.filter((x) => x !== id);
    } else {
      current = [...current, id].slice(-max);
    }
    chips.forEach((c, i) => c.setAttribute("aria-pressed", String(current.includes(methods[i].id))));
    onChange(current);
  }
  container.append(el("span", { class: "eyebrow" }, `Show up to ${max}`), ...chips);
}
