// Data tab: what the models were given. Tables and columns, row counts, gaps, LightGBM feature importance.
import { el, fmtInt, fmtPct, fmtDate, SEGMENT_LABEL } from "./format.js";

const GROUP_COLOR = {
  "Recent sales": "var(--s1)",
  Price: "var(--s2)",
  Calendar: "var(--s3)",
  Events: "var(--s4)",
  "Store / product": "var(--s7)",
};
const TYPE_LABEL = { number: "decimal", integer: "integer", text: "text", date: "date" };

export function renderData(profile) {
  const root = document.getElementById("data-body");
  if (!profile) {
    root.replaceChildren(
      el("div", { class: "card state" }, "No data profile yet. Run: python scripts/run_pipeline.py --stages profile"),
    );
    return;
  }
  root.replaceChildren(
    tiles(profile),
    gapsCard(profile),
    importanceCard(profile.importance),
    ...(profile.raw_files?.length ? [rawFilesCard(profile.raw_files)] : []),
    tablesCard(profile.tables),
  );
}

function rawFilesCard(files) {
  return el("div", { class: "card" }, [
    el("h2", {}, "Raw files"),
    el("p", { class: "muted small" }, "What the M5 competition ships, before any processing. Daily sales are summed to weeks and reshaped from wide (one column per day) to long (one row per series and week)."),
    el("div", { class: "table-wrap" }, el("table", {}, [
      el("thead", {}, el("tr", {}, ["File", "Rows", "Columns", "Holds", "Becomes"].map((h, j) =>
        el("th", { scope: "col", style: j === 0 || j > 2 ? "text-align:left" : "" }, h)))),
      el("tbody", {}, files.map((f) => el("tr", {}, [
        el("td", { class: "mono" }, f.file),
        el("td", {}, fmtInt(f.rows)),
        el("td", {}, fmtInt(f.columns)),
        el("td", { style: "text-align:left;white-space:normal" }, f.holds),
        el("td", { style: "text-align:left;white-space:normal" }, f.becomes),
      ]))),
    ])),
  ]);
}

function tiles({ overview: o, gaps: g }) {
  const items = [
    [fmtInt(o.rows), "rows (series × week)", `${o.weeks} weeks, ${fmtDate(o.first_week)} – ${fmtDate(o.last_week)}`],
    [fmtInt(o.series), "item × store series", `${fmtInt(o.items)} items in ${o.stores} stores, ${o.states} states`],
    [String(o.departments), "departments", "the store / product features"],
    [fmtPct(g.zero_share, 0), "weeks with zero sales", "after each series' first sale"],
    [fmtInt(g.long_run_series), `series with a ${g.long_run_weeks}+ week gap`, `${fmtPct(g.long_run_week_share, 1)} of all rows sit in such gaps`],
  ];
  return el("div", { class: "tiles" }, items.map(([v, k, d]) =>
    el("div", { class: "card tile" }, [el("div", { class: "v" }, v), el("div", { class: "k" }, k), el("div", { class: "d" }, d)])));
}

function gapsCard({ gaps: g, overview: o }) {
  const rows = [
    ["Before the first sale", `${fmtInt(g.late_start_series)} series start after Jan 2011. Their ${fmtInt(g.leading_weeks_trimmed)} weeks before the first sale are dropped, not counted as zero sales.`],
    ["Long zero runs", `${fmtInt(g.long_run_series)} series have at least ${g.long_run_weeks} weeks in a row without a sale after selling before (longest: ${g.long_run_longest} weeks). Likely out of stock or off the shelf; the data cannot tell which. They are kept as zeros.`],
    ["Gone quiet at the end", `${fmtInt(g.long_run_to_end_series)} of those series are still at zero in the last week, probably delisted.`],
    ["Never sold", `${fmtInt(g.never_sold)} series have no sale at all and are left out.`],
    ["Incomplete last week", `The last ${g.tail_days_dropped} days of the raw history do not fill a week and are dropped.`],
    ["Christmas", "Stores are closed on 25 December, so that week is one day short. A calendar flag marks it."],
    [
      "Missing prices",
      g.price_filled_rows === 0
        ? "None. Every row has a listed price, because each series starts at its first sale and items are listed before they sell. A fill-in rule (nearest listed price) stays in place for other data."
        : (g.price_filled_rows != null ? `${fmtInt(g.price_filled_rows)} rows (${fmtPct(g.price_filled_rows / o.rows, 2)}) have no listed price. ` : "") +
          "They take the nearest listed price (forward, then backward), so the price columns below show no missing values.",
    ],
  ];
  const bySeg = Object.entries(g.zero_share_by_segment || {});
  const years = Object.entries(g.start_year);
  const maxYear = Math.max(...years.map(([, n]) => n));
  return el("div", { class: "grid g2 g-wide-left" }, [
    el("div", { class: "card" }, [
      el("h2", {}, "Gaps and how they are handled"),
      el("dl", { class: "kv" }, rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])),
    ]),
    el("div", { class: "card" }, [
      el("h2", {}, "Zero-sales weeks by segment"),
      el("p", { class: "muted small" }, "Share of rows with no sale. Segments as fixed at the test cutoff."),
      bars(bySeg.map(([s, v]) => ({ label: SEGMENT_LABEL[s] || s, value: v, text: fmtPct(v, 0), color: "var(--s1)" })), 1),
      el("h2", { style: "margin-top:16px" }, "When series start selling"),
      el("p", { class: "muted small" }, "Series by year of their first sale."),
      bars(years.map(([y, n]) => ({ label: y, value: n, text: fmtInt(n), color: "var(--s3)" })), maxYear),
    ]),
  ]);
}

function importanceCard(imp) {
  const feats = imp.features;
  const groups = {};
  feats.forEach((f) => { groups[f.group] = (groups[f.group] || 0) + f.gain; });
  const groupRows = Object.entries(groups).sort((a, b) => b[1] - a[1]);
  const top = feats[0]?.gain || 1;
  const w = imp.window;
  const matrix = w.rows
    ? `Training set: ${fmtInt(w.rows)} rows × ${w.features} features from ${fmtInt(w.series)} series. ` +
      `${fmtPct(w.no_lag52_share, 1)} of the rows come from series younger than a year, so their “sales 52 weeks ago” is empty; LightGBM handles the gap natively.`
    : "";
  return el("div", { class: "card" }, [
    el("h2", {}, "What LightGBM relies on"),
    matrix ? el("p", { class: "small", style: "color:var(--text-2)" }, matrix) : null,
    el("p", { class: "muted small" },
      `Level-target model trained at the test cutoff (${fmtDate(imp.window.cutoff)}) on the last ${imp.window.train_weeks} weeks. ` +
      "Bar = share of the total loss reduction from splits on that feature (gain). Split share = how often it is used. " +
      "Correlated features share credit, so a low bar does not mean useless: see the ablation table under Tradeoffs for what removing features costs."),
    el("div", { class: "legend" }, groupRows.map(([g, v]) =>
      el("span", {}, [el("i", { class: "band", style: `--c:${GROUP_COLOR[g] || "var(--s0)"}` }), `${g} ${fmtPct(v, 1)}`]))),
    el("div", { class: "table-wrap" }, el("table", { class: "imp" }, [
      el("thead", {}, el("tr", {}, [
        el("th", { scope: "col" }, "Feature"), el("th", { scope: "col", style: "text-align:left" }, "Gain share"), el("th", { scope: "col" }, "Split share"),
      ])),
      el("tbody", {}, feats.map((f) => el("tr", {}, [
        el("td", { title: f.feature }, [f.label, el("span", { class: "family-tag mono" }, f.feature)]),
        el("td", { class: "bar-cell" }, el("div", { class: "hbar" }, [
          el("i", { style: `width:${(100 * f.gain) / top}%;--c:${GROUP_COLOR[f.group] || "var(--s0)"}` }),
          el("b", {}, fmtPct(f.gain, 1)),
        ])),
        el("td", {}, fmtPct(f.split, 1)),
      ]))),
    ])),
  ]);
}

function tablesCard(tables) {
  return el("div", { class: "card" }, [
    el("h2", {}, "Tables and columns"),
    el("p", { class: "muted small" }, "The processed tables the pipeline builds from the raw M5 files. Click a table to see its columns."),
    ...tables.map((t, i) => el("details", { class: "tbl", open: i === 0 }, [
      el("summary", {}, [el("b", {}, t.title), el("span", { class: "mono muted" }, ` ${t.id}`), el("span", { class: "muted" }, ` · ${fmtInt(t.rows)} rows · ${t.columns.length} columns · ${t.grain}`)]),
      el("div", { class: "table-wrap" }, el("table", {}, [
        el("thead", {}, el("tr", {}, ["Column", "Meaning", "Type", "Distinct", "Missing", "Min", "Max"].map((h, j) =>
          el("th", { scope: "col", style: j < 3 ? "text-align:left" : "" }, h)))),
        el("tbody", {}, t.columns.map((c) => el("tr", {}, [
          el("td", { class: "mono" }, c.name),
          el("td", { style: "text-align:left;white-space:normal" }, c.meaning || "–"),
          el("td", { style: "text-align:left" }, TYPE_LABEL[c.type] || c.type),
          el("td", {}, fmtInt(c.distinct)),
          el("td", { class: c.missing_share > 0 ? "warn" : "" }, c.missing_share > 0 ? fmtPct(c.missing_share, 2) : "none"),
          el("td", {}, fmtVal(c.min)),
          el("td", {}, fmtVal(c.max)),
        ]))),
      ])),
    ])),
  ]);
}

function bars(rows, max) {
  return el("div", { class: "hbars" }, rows.map((r) => el("div", { class: "hbar-row" }, [
    el("span", {}, r.label),
    el("div", { class: "hbar" }, [el("i", { style: `width:${(100 * r.value) / (max || 1)}%;--c:${r.color}` }), el("b", {}, r.text)]),
  ])));
}

const fmtVal = (v) => (v === undefined || v === null ? "–" : typeof v === "number" ? (Number.isInteger(v) ? fmtInt(v) : v.toFixed(2)) : fmtDate(v));
