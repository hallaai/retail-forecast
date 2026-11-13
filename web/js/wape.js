// Interactive WAPE / MAPE / MASE example with three items and editable forecasts.
import { el, fmtPct, fmtNum } from "./format.js";

const ITEMS = [
  { name: "Hit (milk)", actual: 200, forecast: 180, naiveMae: 25 },
  { name: "Regular (pasta)", actual: 20, forecast: 24, naiveMae: 5 },
  { name: "Slow seller (spare part)", actual: 0, forecast: 1, naiveMae: 0.8 },
];

export function renderWapeExample(table, out) {
  const state = ITEMS.map((i) => ({ ...i }));
  const inputs = state.map((item, k) =>
    el("input", {
      type: "number", min: "0", step: "1", value: String(item.forecast), id: `wape-f-${k}`,
      "aria-label": `Forecast for ${item.name}`,
      oninput: (ev) => {
        const v = Number(ev.target.value);
        state[k] = { ...state[k], forecast: Number.isFinite(v) && v >= 0 ? v : 0 };
        update();
      },
    }),
  );
  const errCells = state.map(() => el("td"));
  table.replaceChildren(
    el("thead", {}, el("tr", {}, [el("th", {}, "Item"), el("th", {}, "Actual"), el("th", {}, "Forecast"), el("th", {}, "|Error|")])),
    el("tbody", {}, state.map((item, k) => el("tr", {}, [el("td", {}, item.name), el("td", {}, String(item.actual)), el("td", {}, inputs[k]), errCells[k]]))),
  );

  function update() {
    const absErr = state.map((i) => Math.abs(i.actual - i.forecast));
    absErr.forEach((e, k) => { errCells[k].textContent = fmtNum(e, 0); });
    const wape = absErr.reduce((a, b) => a + b, 0) / state.reduce((a, i) => a + i.actual, 0);
    const apes = state.map((i, k) => (i.actual === 0 ? Infinity : absErr[k] / i.actual));
    const mape = apes.reduce((a, b) => a + b, 0) / apes.length;
    const mase = state.reduce((a, i, k) => a + absErr[k] / i.naiveMae, 0) / state.length;
    out.replaceChildren(
      el("div", {}, [el("b", {}, fmtPct(wape)), el("span", {}, "WAPE: led by the hit")]),
      el("div", {}, [el("b", {}, Number.isFinite(mape) ? fmtPct(mape) : "∞"), el("span", {}, "MAPE: breaks on a zero week")]),
      el("div", {}, [el("b", {}, fmtNum(mase)), el("span", {}, "MASE: each item counts equally")]),
    );
  }
  update();
}
