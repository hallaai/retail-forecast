# M5 Forecast Bench

Weekly demand forecasting on the Walmart M5 data: 30,490 item × store series, 12 weeks ahead.
The study compares five statistical baselines, a global LightGBM (two target designs, two
ablations), a zero-shot foundation model (Chronos-Bolt) and an ensemble, and scores them by
segment: hits, middle, sparse sellers and new items. The results are shown on a local web page
(dark mode first) served by a small FastAPI app.

Contents: [Results](#results) · [Findings](#three-findings) · [What WAPE is](#what-wape-is) ·
[Tradeoffs](#tradeoffs) · [How to run](#how-to-run) · [Project structure](#project-structure) ·
[Pipeline](#how-the-pipeline-works) · [Limitations](#limitations)

---

## Setup in one paragraph

Daily M5 sales are summed to Saturday–Friday weeks: 277 complete weeks, Jan 2011 – May 2016.
Each series starts at its first sale (dense grid, zero weeks kept). Rolling origin: four 12-week
backtest windows (Mar 2015 – Feb 2016) and one test window (27 Feb – 14 May 2016). Every model
choice, including which models form the ensemble, is made on the backtests; the test window is
only scored. Segments are assigned at each forecast origin from history only:

| Segment | Rule | Series at test cutoff | Share of units |
|---|---|---:|---:|
| New | fewer than 52 weeks since first sale | 573 | 1% |
| Sparse | 30% or more zero weeks in the last 52 | 9,871 | 11% |
| Hit | top 10% of the remaining series by 52-week volume | 2,006 | 41% |
| Middle | everything else | 18,040 | 47% |

## Results

Test window WAPE by method and segment (backtest mean in brackets). Lower is better.

| Method | All | Hits | Middle | Sparse | New |
|---|---:|---:|---:|---:|---:|
| Naive | 48.5% (54.5) | 35.2% | 52.2% | 68.2% | 56.8% |
| Seasonal naive (52) | 62.2% (66.4) | 48.0% | 63.0% | 92.4% | 58.4% |
| AutoETS | 43.3% (48.6) | 32.2% | 44.8% | 63.8% | 52.7% |
| Croston SBA | 44.6% (51.4) | 32.4% | 44.5% | 72.2% | 47.4% |
| TSB | 43.5% (48.8) | 31.9% | 44.1% | 67.8% | 48.2% |
| LightGBM, level target | 41.1% (46.4) | 31.3% | **42.7%** | 58.4% | 47.1% |
| LightGBM, ratio target | 41.1% (47.0) | 30.8% | 43.1% | **58.4%** | **46.9%** |
| Chronos-Bolt (zero-shot) | 42.7% (47.6) | 31.9% | 44.4% | 61.8% | 49.4% |
| Ensemble (LightGBM level + Chronos) | **40.9%** (**46.0**) | **30.5%** | 42.8% | 58.5% | 47.1% |

| Method | Bias all | Bias sparse | Pinball all | 80% coverage | WAPE range over 5 windows |
|---|---:|---:|---:|---:|---:|
| Naive | −6.1% | −20.6% | 35.4% | 96% | 48.5–58.3% |
| Seasonal naive (52) | −12.7% | −43.3% | 38.7% | 82% | 62.2–70.2% |
| AutoETS | −3.7% | −20.2% | 28.7% | 83% | 43.3–51.7% |
| Croston SBA (calibrated) | −11.0% | −31.8% | 30.5% | 82% | 44.6–52.4% |
| TSB (calibrated) | −8.0% | −34.4% | 29.7% | 81% | 43.5–49.9% |
| LightGBM, level (calibrated) | −4.2% | −11.6% | 28.1% | 81% | 41.1–47.7% |
| LightGBM, ratio (calibrated) | −2.5% | −14.7% | 28.3% | 81% | 41.1–48.5% |
| Chronos-Bolt | −6.1% | −25.7% | 27.8% | 80% | 42.7–49.4% |
| Ensemble | −5.2% | −18.7% | **27.0%** | 80% | 40.9–47.6% |

Side experiments (test, backtest mean in brackets):

| Variant | WAPE all | WAPE sparse | Bias all |
|---|---:|---:|---:|
| LightGBM level, all features | 41.1% (46.4) | 58.4% | −4.2% |
| … without price / calendar features | 41.4% (47.3) | 59.0% | −4.8% |
| … prices frozen at the forecast origin | 41.4% (46.8) | 59.3% | −4.2% |
| Chronos, median as point (its built-in output) | 42.8% (47.5) | 63.2% | −10.7% |
| Chronos, mean of 9 deciles as point (used) | 42.7% (47.6) | 61.8% | −6.1% |

Median runtime for one 12-week forecast of all 30,490 series (12-core ARM laptop CPU, no GPU,
median of 5 runs): TSB / Croston 13 s, Naive 17 s, Seasonal naive 33 s, AutoETS 73 s,
LightGBM 74–77 s (features + training + forecast), Chronos-Bolt 215 s, ensemble ~290 s.
Chronos-2 was measured 8.5× slower than Chronos-Bolt-small and was not run.

All scores (every method × window × segment × metric) are in `data/results/full/scores.csv`.

## Three findings

1. **A global LightGBM is the best single model, and the gain comes from learning across series.**
   Test WAPE 41.1% against 43.3% for AutoETS and 62.2% for seasonal naive; best single model in all
   five windows. Dropping every price and calendar feature costs only 0.3 points on test (0.9 on
   backtests), and freezing prices at the forecast origin costs 0.3. Most of the ~2-point gain is one
   model learning demand shapes from 30k series' lags, which a per-series model cannot do.
2. **Zero-shot Chronos gives the best ranges, and its point definition matters.** With no training
   or features it matches AutoETS on WAPE and has the best pinball loss of any single model in every
   window. Its built-in point output is the median: same WAPE, but bias −10.7% against −6.1% with
   the mean of its deciles. The ensemble with LightGBM (pair chosen on backtests) has the best
   pinball in all five windows (27.0% on test) but only ties LightGBM on WAPE (gaps 0.1–0.8 points).
3. **Sparse items have no stable winner, and a selection effect makes everyone look low.** The best
   sparse method changes from window to window (AutoETS, Chronos, the ensemble, LightGBM). Items
   selected for recent zero weeks sell 30–68% more in the next 12 weeks than in their trailing year:
   regression to the mean, helped by stock-outs recorded as zero demand. LightGBM is close to
   unbiased on sparse items in the backtests (−3%) but −12% on test, where that jump was largest;
   Chronos and TSB are low in every window. For the long tail, decide on calibrated quantiles and
   watch bias, not WAPE alone.

## What WAPE is

```
WAPE = Σ |actual − forecast| / Σ actual        (over every scored item-week)
```

* Units mis-forecast per unit sold. WAPE 0.40 = 40 wrong units per 100 sold; `1 − WAPE` is often
  quoted as "forecast accuracy".
* Works with zero weeks. MAPE divides by each week's actual and breaks on zero sales.
* Volume-weighted. Hits dominate; it measures business impact, not the average series.
* Minimised by the median. On skewed, zero-heavy demand a forecast that runs low can still score
  well (see the Chronos median vs mean row). Always report bias next to it.

Companion metrics used here:

| Metric | Formula | Answers |
|---|---|---|
| Bias | Σ(forecast − actual) / Σ actual | Systematic over (+) or under (−) forecasting |
| MASE | mean over series of MAE ÷ in-sample naive MAE | The typical series; every item counts equally |
| Pinball | mean over q ∈ {0.1, 0.5, 0.9} of Σ pinball_q / Σ actual, ×2 | Quality of the whole range (×2 so the median term alone equals WAPE) |
| 80% coverage | share of actuals inside [q10, q90] | Whether ranges are honest; target 80% |

## Tradeoffs

| Approach | Good at | Weak at | Use when |
|---|---|---|---|
| Local statistical (Naive, ETS, Croston, TSB) | Seconds, nothing to train, explainable per item | Blind to price/events, learns nothing across items | Benchmark and fallback |
| Global LightGBM | Best single model; learns across 30k series; can use drivers | Feature pipeline and retraining; recursive errors compound; ranges need calibration | Production default for retail |
| Chronos zero-shot | No training; best native ranges | Median point runs low; no covariates here; ~3× LightGBM's CPU time; hard to explain | Fast strong baseline, ranges without calibration, ensemble member |
| Ensemble (LightGBM + Chronos) | Most dependable ranges, lowest WAPE in all 5 windows | WAPE gain within noise; two pipelines, ~4× compute | Range quality drives the decision (safety stock) |
| Ratio target (y ÷ trailing mean) | One scale for all items | No gain here: level wins WAPE in all 4 backtests and pinball in all 5 windows | Test it, don't assume it helps |
| Seasonal ETS (m = 52) | Yearly seasonality | 12× slower (0.32 s vs 0.027 s per series), 52 states from ~5 cycles | Dropped |

---

## How to run

Tested on Windows 11 (ARM64), Python 3.13. Commands below are for PowerShell; on macOS/Linux use
`.venv/bin/python` instead of `.venv\Scripts\python.exe`.

### 1. Environment

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
# behind a corporate TLS proxy add: --system-certs
```

(Without uv: `python -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`.)

### 2. Data (not in the repo)

Put `sales_train_evaluation.csv`, `sell_prices.csv` and `calendar.csv` in `data/raw/`:

* Kaggle: competition "M5 Forecasting - Accuracy" (accept the rules, download the files), or
* Nixtla's public mirror: `https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip`

```powershell
mkdir data\raw
curl.exe -L -o data\raw\m5.zip https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip
tar -xf data\raw\m5.zip -C data\raw sales_train_evaluation.csv sell_prices.csv calendar.csv
```

The first run downloads the Chronos-Bolt-small weights (~200 MB) from Hugging Face.

### 3. Pipeline

```powershell
# quick dev run on 600 random series (~15 min)
.venv\Scripts\python.exe scripts\run_pipeline.py --sample 600

# full run: all 30,490 series, 5 windows, 10 models + ensemble (~60 min on a 12-core CPU)
.venv\Scripts\python.exe scripts\run_pipeline.py

# re-score and re-export only, from cached forecasts (~4 min)
.venv\Scripts\python.exe scripts\run_pipeline.py --stages score,export
```

Stages: `prepare` (weekly panel, runs automatically if missing) → `forecast` (cached per method ×
window in `data/forecasts/<run>/`; delete a folder to recompute it) → `score` → `export` (writes
`web/data/*.json`) → `profile` (writes `web/data/profile.json` for the Data tab: table and column
profile, raw-file sizes, gaps incl. imputed prices, LightGBM training-set size and feature importance from one
extra fit at the test cutoff, ~3 min, cached in `data/results/<run>/lgbm_importance.csv`; delete that file to refit). Note: a `--sample` run also exports, so it overwrites `web/data`; run the full
pipeline (or `--stages score,export`) again before viewing the full results.

### 4. Web page

```powershell
.venv\Scripts\python.exe -m uvicorn server.app:app --port 8050
```

Open http://localhost:8050 and stop with Ctrl+C. Tabs: Overview, Results, WAPE & metrics, Tradeoffs,
Explorer, Data, Method. The page reads only `web/data/*.json`, which are committed, so it works right
after cloning. The series search box in the Explorer needs `data/results/full/test_forecasts.parquet`
and `data/processed/`, which a full pipeline run creates.

API (used by the Explorer search):

| Endpoint | Returns |
|---|---|
| `GET /api/health` | `{"status": "ok"}` |
| `GET /api/series?q=FOODS_3&limit=20` | matching series ids with their segment |
| `GET /api/series/{id}` | 2 years of history, 12 test weeks, every method's forecast, 10–90% range and WAPE |

### 5. Tests

```powershell
.venv\Scripts\python.exe -m pytest -q tests                      # 50 tests, ~1.5 min, no M5 files needed
.venv\Scripts\python.exe -m pytest -q tests --cov=src/m5demo --cov=server
$env:RUN_CHRONOS = "1"; .venv\Scripts\python.exe -m pytest -q tests  # also tests the Chronos adapter
.venv\Scripts\ruff.exe check --line-length 130 src server scripts tests
```

---

## Project structure

```
Retail-forecast/
├── README.md               this file
├── CHANGELOG.md
├── requirements.txt        pinned versions
├── .gitignore
├── src/m5demo/             the pipeline
│   ├── config.py           every constant: horizon, windows, segment rules, LightGBM params
│   ├── data.py             raw CSV -> weekly dense panel (the only module that reads raw files)
│   ├── splits.py           rolling-origin windows                          (pure)
│   ├── segments.py         hit / middle / sparse / new from pre-cutoff data (pure)
│   ├── features.py         price + calendar features, ratio target, price freezing (pure)
│   ├── metrics.py          WAPE, bias, MASE, pinball, coverage             (pure)
│   ├── calibration.py      backtest-residual quantiles for point-only methods (pure)
│   ├── forecast_frame.py   the one forecast schema (f, q10, q50, q90), clipping, sanitising
│   ├── models_stats.py     StatsForecast: Naive, SeasonalNaive(52), AutoETS, CrostonSBA, TSB
│   ├── models_lgbm.py      MLForecast + LightGBM: level, ratio, and two ablations
│   ├── models_chronos.py   Chronos-Bolt-small zero-shot (point = mean of 9 deciles)
│   ├── pipeline.py         orchestration, forecast cache + timings, calibration, ensemble, scoring
│   ├── evaluate.py         join forecasts with actuals, score per segment
│   ├── profile.py          data profile for the Data tab: columns, rows, gaps, feature labels (pure)
│   └── export.py           JSON for the page (aggregates + 100 sample series, no raw data)
├── scripts/run_pipeline.py end-to-end CLI
├── server/app.py           FastAPI: serves web/ and the series lookup API
├── web/                    the page (vanilla JS + SVG, no build step)
│   ├── index.html          layout, styles (dark-first tokens), all tabs
│   ├── js/                 app (router, theme), tables, charts, explorer, WAPE widget
│   └── data/               summary.json, explorer.json, profile.json (pipeline output), study.json (text)
├── tests/                  unit, engine (synthetic data), adapters, leakage, API tests
└── data/                   (mostly git-ignored)
    ├── raw/                M5 CSVs                                   ignored
    ├── processed/          weekly panel, features (parquet)          ignored, regenerable
    ├── forecasts/full/     cached forecasts per method/window + runtimes.json   ignored
    └── results/full/       scores.csv (kept), test_forecasts.parquet (ignored, 60 MB)
```

## How the pipeline works

* **Time.** One period = one Walmart week (Saturday–Friday), labelled by its Saturday. `week` is a
  0-based index; cutoff `c` means a model sees weeks ≤ c and forecasts c+1..c+12. The final 2-day
  partial week is dropped, never fitted.
* **No look-ahead.** Training rows, segments, MASE scales, ratio scales and calibration all use data
  up to the cutoff only. Tests in `tests/test_leakage.py` poison future actuals and check that the
  forecasts do not change.
* **Statistical models** (StatsForecast): Naive, SeasonalNaive(52) (falls back to Naive for series
  under one year), AutoETS (non-seasonal, see Tradeoffs), CrostonSBA, TSB(0.1, 0.1).
* **LightGBM** (MLForecast, Tweedie loss, one global model): lags 1, 2, 4, 52; rolling means 4, 13,
  52; price, discount vs 52-week max, price vs department, SNAP days, event types, Thanksgiving /
  Christmas / Super Bowl / Easter flags, week of year, department / category / store / state. Trained
  on the last 156 weeks, recursive 12-step forecast. Variants: level target; ratio target (units ÷
  trailing 26-week mean); ablations without price/calendar features and with prices frozen at the
  cutoff.
* **Prices.** Future prices are treated as known in advance, like planned prices and markdowns.
  M5 gives realised shelf prices, so this is the standard M5 assumption; the frozen-price ablation
  shows it is worth 0.3 points.
* **Chronos-Bolt-small**: zero-shot, univariate. Its own "mean" output is the median, so the point
  forecast is the mean of its 9 deciles; ranges are its 10th / 90th percentiles.
* **Ensemble**: mean of the two of {LightGBM level, LightGBM ratio, Chronos} with the lowest mean
  backtest WAPE. Picked automatically: LightGBM level + Chronos.
* **Ranges for point-only methods** (Croston, TSB, LightGBM): empirical quantiles of scaled
  residuals (÷ trailing mean) per segment × horizon step from the backtest windows. Each backtest
  window is calibrated on the other three; the test window on all four.
* **Clean-up**: all forecasts clipped at 0; a non-finite interval (AutoETS on one constant series)
  is replaced by the point forecast.

## Limitations

* One retailer, one test season (spring 2016). Differences under about 1 WAPE point are within
  window-to-window noise; several segment-level gaps hold on test but not in every backtest.
* Stock-outs are not modelled: a zero week can be an empty shelf, so history under-states demand for
  sparse items.
* Segments regress to the mean (sparse items sell 30–68% more after selection), which by itself makes
  models look biased low on them.
* Unit-weighted WAPE; a value-weighted version (units × price) could change the ranking.
* No LightGBM tuning; Chronos without covariates; cold-start items (no sale before a cutoff) excluded.
* Backtest coverage of calibrated methods is slightly optimistic (leave-one-window-out uses later
  windows); test-window ranges use earlier windows only.
* Runtimes come from one laptop; a few runs shared the CPU with other work, hence medians.

An independent review of the pipeline found no look-ahead leakage. It did correct four things in a
first draft: the Chronos point definition, an ensemble picked with test-window knowledge,
unsupported feature-importance claims, and segment claims that held on test only. All four are fixed
(see CHANGELOG.md).

## Putting it on GitHub

```powershell
git init
git add .
git commit -m "feat: M5 weekly forecasting study with web demo"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

`.gitignore` keeps out the virtual environment, the M5 raw data (Kaggle's competition rules don't
allow redistribution) and the regenerable intermediates (~230 MB). The committed `scores.csv` and
`web/data/*.json` are enough to view the page.
