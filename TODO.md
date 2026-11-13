# TODO

## Ensemble (model fusion)

### Current state

The ensemble is a plain 50/50 average of LightGBM (level target) and Chronos-Bolt
([src/m5demo/pipeline.py](src/m5demo/pipeline.py), `with_ensemble` and `_ensemble`).

- Candidates: `lgbm_level`, `lgbm_ratio`, `chronos_bolt` (`ENSEMBLE_CANDIDATES`).
- All three pairs are tried. The pair with the lowest mean WAPE over the 4 backtest windows is
  kept (the test window is never consulted). LightGBM level + Chronos won.
- Per series and week, the point forecast and the q10, q50 and q90 quantiles are each averaged
  with equal weights.

Result on the test window: best pinball loss (27.0%), well-calibrated 80% ranges. The WAPE gain
over LightGBM alone (40.9% vs 41.1%) is within noise. Averaging mostly improves the reliability
of the ranges, because the two models make different kinds of errors.

### Limitations

- [ ] Fixed equal weights: the same blend for every series, segment and forecast week, although
      the best model differs by segment (for sparse series it changes from window to window).
- [ ] Averaging two models' q10/q90 is a simple approximation, not a proper mixture of the two
      forecast distributions.
- [ ] Only pairs of three models are considered. The statistical models (AutoETS, TSB) are never
      candidates.

### Upgrades (roughly from simplest to most complex)

All weights and choices must be fitted on the backtest windows only, never on the test window.

- [ ] **Tuned weight**: choose `w` in `w·LGBM + (1−w)·Chronos` on the backtests, for example on
      a grid from 0 to 1 in steps of 0.1.
- [ ] **Weights per segment**: separate weights for hits, middle, sparse and new items, and allow
      TSB or AutoETS as a member for sparse series. This is the suggested next step.
- [ ] **Inverse-error weights**: weight each model by 1/WAPE from recent backtest windows.
- [ ] **Stacking**: a small model learns the blend from backtest forecasts plus features
      (segment, horizon, zero share). Higher risk of overfitting; check it with leave-one-window-out
      across the backtests.
- [ ] **Quantile fusion**: combine the quantiles by minimising pinball loss directly, instead of
      a plain average.

For each upgrade: add it as a new ensemble variant next to the current one, add unit tests
(weights sum to 1, quantiles stay ordered, no test-window leakage), and report it in the results
table and the web page.

## Cold start

### Current state

There are two kinds of cold start, and the demo handles only the first.

**1. Young series (some history, less than a year).** These are handled, and they are scored
as their own segment, `new`.

- A series enters the data at its first sale ([src/m5demo/data.py](src/m5demo/data.py)). Weeks
  before the first sale are dropped, not counted as zero sales.
- At each cutoff, a series younger than 52 weeks is in the `new` segment
  ([src/m5demo/segments.py](src/m5demo/segments.py)). The rule uses only history up to the cutoff.
  This segment has 2,521 series in bt1, falling to 573 in test (1.4% of test volume).
- How each method copes with a short history:
  - **Naive, Croston, TSB:** work from a single week of history.
  - **SeasonalNaive:** has no value from 52 weeks ago, so it falls back to Naive
    (`_fallback_to_naive` in [src/m5demo/models_stats.py](src/m5demo/models_stats.py)).
  - **AutoETS:** non-seasonal, so a short series is enough. If a fit fails, StatsForecast uses
    `fallback_model=Naive()`.
  - **LightGBM:** one global model across all series. For a young series, lag 52 and the 52-week
    mean are missing (NaN), and so are the 13-week and 4-week means until those windows fill.
    LightGBM routes the NaN to a learned branch, so no values are imputed. The store, department
    and category features and the price features let the model borrow from similar series. 11.6%
    of training rows have no lag 52.
  - **Chronos:** a univariate zero-shot model, so a context of any length works.
- Calibration: point-only methods get their 80% ranges from backtest residuals, per
  (segment, horizon week), so `new` gets its own, usually wider, ranges
  ([src/m5demo/calibration.py](src/m5demo/calibration.py)).
- Result on the `new` segment, test window: LightGBM 47.1% WAPE (bias −1.3%), ensemble 47.1%,
  Chronos 49.4%, AutoETS 52.7%, SeasonalNaive 58.4%. This is worse than the overall 40.9%, as
  expected.

**2. True cold start (no sale before the cutoff).** This is out of scope.

- `train_part` and `actuals_part` ([src/m5demo/splits.py](src/m5demo/splits.py)) keep only
  series with at least one sale up to the cutoff. A series whose first sale falls inside the
  horizon is never forecast and never scored.
- The cost is small here: 365 series in bt1 (0.29% of horizon volume), 66, 67 and 27 in bt2 to
  bt4, and none in test. M5 has no launches near the end of the data.

### Limitations

- [ ] A product launch cannot be forecast at all. In practice that is often the question
      ("we list a new item next month, how much do we order?").
- [ ] There is no explicit age feature. LightGBM infers youth only from which lags are missing,
      so it cannot learn a ramp-up curve (weeks since launch → share of mature volume).
- [ ] `item_id` is not a feature, so a new store for an existing item cannot borrow that item's
      history from other stores.
- [ ] `new` is a one-year bucket. Weeks 1–4 after launch behave very differently from weeks
      40–52 and are not reported separately.
- [ ] The excluded series introduce a small selection effect: scores cover only series that
      already existed at the cutoff.

### Upgrades (roughly from simplest to most complex)

- [ ] **Report by age**: split `new` into 0–4, 5–12, 13–26 and 27–51 weeks, and show WAPE and
      bias per bucket on the Results page.
- [ ] **Age feature**: add `weeks_since_first_sale` (capped at 52) to LightGBM, so it can learn
      the ramp-up.
- [ ] **Item-level borrowing**: add item mean sales across other stores (lagged, up to the cutoff
      only) as a feature, so a new item × store starts from what the item sells elsewhere.
- [ ] **Analog forecast for true cold start**: for a series with no sales, forecast from the
      attributes known before launch (department, store, price relative to the department).
      Either average the first 12 weeks of similar past launches, or train a small model on past
      launches. Score it on the series that bt1 to bt4 currently exclude.
- [ ] **Launch curve**: fit a typical weeks-since-launch curve per department and scale it by
      an analog volume. Blend it with the series' own history as weeks accumulate.

For each upgrade: launches must be split by time, never at random (the model only learns from
launches before the cutoff). Add a leakage test that perturbs post-cutoff sales of the scored
launches and checks the forecast does not change.
