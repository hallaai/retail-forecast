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
