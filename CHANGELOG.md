# Changelog

## 2026-11-13 (later)

- Data tab: row and series counts, every processed table's columns (meaning, type, distinct, missing, range),
  gaps (late starts, 13+ week zero runs, delisted-looking series, never-sold series, dropped tail days) and
  LightGBM (level) gain / split feature importance at the test cutoff.
- New `profile` pipeline stage and `src/m5demo/profile.py`; `models_lgbm.feature_importance` shares the
  training code with `forecast`, so importance describes the model that forecasts.
- Data tab also shows the raw M5 files (rows, columns, what each becomes), rows with an imputed price (none in M5:
  series start at their first sale) and the LightGBM training-set size (4.37M rows × 25 features, 11.6% without lag 52).

## 2026-11-13

- Weekly M5 pipeline: dense panel, rolling-origin windows (4 backtest + test, 12 weeks), segments per cutoff.
- Methods: StatsForecast baselines, MLForecast + LightGBM (level and ratio target), Chronos-Bolt zero-shot, ensemble.
- Metrics: WAPE, bias, MASE, scaled pinball, 80% coverage; backtest-residual calibration for point-only methods.
- Forecast cache per method × window with timed runs; JSON export for the frontend.
- FastAPI server (static study page + series lookup) and a dark-first study page in the Food Recs Lab style.
- Tests: unit, engine on synthetic data, adapters, API integration (86% line coverage).

### After independent review (same day)

- Chronos point forecast = mean of its 9 deciles (its built-in "mean" is the median); median kept as a side variant.
- Ensemble pair chosen automatically on backtest WAPE (now LightGBM level + Chronos); test never consulted.
- LightGBM ablations: no price / calendar features, prices frozen at the cutoff.
- Non-finite AutoETS intervals replaced by the point forecast; ensemble members must be row-aligned.
- Regression tests: preprocess row alignment, future-actuals perturbation, ensemble choice (87% coverage, 50 tests).
- Findings rewritten to what holds across windows; stock-out censoring and selection effect added to limitations.
- Frontend: Food Recs Lab style, dark-first, tabs; stability by segment, ablation and median-vs-mean tables.
