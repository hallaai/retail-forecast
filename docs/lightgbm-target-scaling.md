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

RESULTS_TABLE

## How to run one variant or change one

* All variants are defined in `SPECS` in `src/m5demo/models_lgbm.py`: target, objective, weights,
  exogenous features. Add a `Spec` there and to `pipeline.LGBM_SCALING`, and it is trained, cached,
  calibrated, scored and shown like the others.
* Forecasts are cached per method × window in `data/forecasts/full/<method>/`. Delete one folder
  to retrain that variant; `python scripts/run_pipeline.py` retrains only what is missing.
* The ensemble still chooses only between `lgbm_level`, `lgbm_ratio` and Chronos, so these
  experiments do not change the headline results.
