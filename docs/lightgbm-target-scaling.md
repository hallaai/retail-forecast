# LightGBM: scale the target or not

One global LightGBM forecasts all 30,490 item × store series. Their sizes range from a few units a
year to hundreds a week, so the model has to cope with very different levels. There are three ways
to handle that, and this project now runs all of them side by side on the same windows and series.

## The options

| Variant (method id) | Target | Loss | How the level is handled | Back to units |
|---|---|---|---|---|
| `lgbm_level` | weekly units | Tweedie (power 1.1) | Recent-sales lags (1, 2, 4, 52 weeks; 4/13/52-week means) anchor each series. Tweedie's log link makes effects multiplicative, so "a 30% discount → ×1.8" applies to a slow seller and a hit alike. | nothing to undo |
| `lgbm_ratio` | units ÷ (series' trailing 26-week mean + 0.1) | Tweedie | Explicit: every series is on a "1 = normal week" scale. | × the same scale |
| `lgbm_log1p` | log(1 + units) | squared error | The log compresses big and small series onto one scale. | expm1, then × a smearing factor (see below) |
| `lgbm_local_scaled` | (units − series mean) ÷ series std (`LocalStandardScaler`) | squared error | Explicit per-series standardising. | MLForecast inverts it inside `predict` |
| `lgbm_level_weighted` | weekly units | Tweedie, rows weighted by the series' trailing 52-week mean volume | Like `lgbm_level`, but big sellers count more in training. This is not a target transform; it moves training toward WAPE, where every unit sold counts the same. | nothing to undo |

The first is what Zalando describes for its sales-event pricing model: raw counts, Tweedie loss, and
recent-history features. Its pricing transformer puts sales on a log scale instead, which is the
`lgbm_log1p` idea.

## Watch out: `LocalStandardScaler` + Tweedie fails

A commonly copied MLForecast snippet looks like this:

```python
MLForecast(
    models=[lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3)],
    target_transforms=[LocalStandardScaler()],   # or Differences([1])
    ...
)
```

It fails. Standard scaling subtracts each series' mean, so about half of the scaled targets are
negative, and `Differences([1])` produces negative week-on-week changes. Tweedie, Poisson and
Gamma model a log of the expected value and accept only targets of zero or more (Gamma: above
zero). LightGBM stops with `[tweedie]: at least one target label is negative` (checked with
LightGBM in this project's environment, same for `poisson` and `gamma`).

Rules this project follows (enforced by `models_lgbm.validate`, which rejects the combination
with an explanation before any training starts):

* Tweedie / Poisson / Gamma: only targets that stay ≥ 0: raw units, a ratio to a mean, `log1p`.
* Standard scaling or differencing: use `objective="regression"` (squared error) or another
  loss that accepts any real number.

## Two details that matter

**Lags are built on the transformed scale.** `log1p` and per-series scaling are MLForecast
`target_transforms`, so the transform happens before lags are computed and is inverted inside the
recursive `predict`. Applying `log1p` to `y` by hand and fitting would feed log-scale predictions
back into raw-scale lags from week 2 onward. The ratio target is applied outside MLForecast, which
is safe because it is one constant per series.

**`expm1` of a log forecast runs low.** Squared error on the log scale predicts the mean of
log(1 + y), and converting that back gives roughly the median, not the mean in units. Forecasts
would be biased low, and the bias metric would show it. `lgbm_log1p` multiplies (1 + forecast) by
Duan's smearing factor, the mean of exp(residual) on the training rows. In-sample residuals are a
little too small, so the correction is slightly too small too.

**Weighting slows small series down.** With volume weights, a small series contributes small
gradients, so it takes more boosting rounds to reach its own level. With the production 400 trees
the weighted and unweighted levels match on synthetic data; with 20 trees the weighted model
over-forecast small series about 6×. Tune the tree count again if you use weights.

## Results (full run, all series)

Test window, backtest mean of 4 windows in brackets. Lower is better except coverage (target 80%).

| Variant | WAPE all | WAPE hits | WAPE middle | WAPE sparse | Bias | Pinball | 80% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lgbm_level` (no scaling) | **41.1%** (**46.4**) | 31.3% (**35.1**) | **42.7%** (**46.3**) | **58.4%** (75.9) | −4.2% (−1.9) | **28.1%** (**31.5**) | 81.2% (79.9) |
| `lgbm_ratio` | **41.1%** (47.0) | **30.8%** (36.5) | 43.1% (46.9) | **58.4%** (**74.1**) | −2.5% (−0.8) | 28.3% (31.8) | 80.6% (79.9) |
| `lgbm_log1p` | 43.7% (47.3) | 35.3% (37.5) | 43.3% (46.4) | 63.2% (74.7) | −11.2% (−9.6) | 29.3% (31.8) | 81.1% (79.9) |
| `lgbm_local_scaled` | 41.7% (47.5) | 31.3% (36.6) | 43.7% (47.4) | 59.6% (76.5) | −1.6% (+2.1) | 28.5% (32.1) | 80.8% (79.9) |
| `lgbm_level_weighted` | 41.7% (47.6) | 31.2% (**35.1**) | 43.3% (47.1) | 60.3% (80.7) | **+0.3%** (+3.0) | 28.6% (31.9) | 81.3% (79.9) |

WAPE, all series, per backtest window:

| Window | level | ratio | log1p | local_scaled | weighted |
|---|---:|---:|---:|---:|---:|
| BT1 | **47.7%** | 48.5% | 48.4% | 49.3% | 49.0% |
| BT2 | **45.9%** | 46.7% | 46.3% | 47.1% | 47.1% |
| BT3 | **46.6%** | 47.1% | 47.2% | 47.8% | 47.8% |
| BT4 | **45.5%** | 45.7% | 47.2% | 45.9% | 46.6% |

What this says:

1. **No scaling wins.** The unscaled Tweedie model has the lowest WAPE in all four backtests and the
   best pinball. Tweedie's log link plus recent-sales lags already handle the level; extra scaling
   adds nothing on M5. The ratio target ties on the test window and is the most even across segments.
2. **log1p is the worst, and biased low even after smearing** (−11% on test). The smearing factor
   was 1.18 in every window, but squared error on the log scale still fits a median-like forecast,
   and sparse items (many zeros) pull it down hardest. Use it only with a stronger, validated
   bias correction.
3. **Per-series standardising and volume weights trade 0.5–1 WAPE point for less bias.** Both move
   bias close to zero (−1.6% and +0.3% on test against −4.2%). If under-forecasting costs more than
   the extra error (empty shelves), that trade can be worth it. Volume weights hurt sparse items
   most (80.7% backtest WAPE), as expected: they get the least weight.
4. **Compute.** log1p is about 3× slower per fit (~4 min vs ~1.3 min), mostly the extra pass over
   4.3M training rows for the smearing factor. The other two cost the same as `lgbm_level`.

These are point forecasts with the same backtest-residual calibration for the ranges, which is why
80% coverage is near 80% for every variant.

## How to run one variant or change one

* All variants are defined in `SPECS` in `src/m5demo/models_lgbm.py`: target, objective, weights,
  exogenous features. Add a `Spec` there and to `pipeline.LGBM_SCALING`, and it is trained, cached,
  calibrated, scored and shown like the others.
* Forecasts are cached per method × window in `data/forecasts/full/<method>/`. Delete one folder
  to retrain that variant; `python scripts/run_pipeline.py` retrains only what is missing.
* The ensemble still chooses only between `lgbm_level`, `lgbm_ratio` and Chronos, so these
  experiments do not change the headline results.
