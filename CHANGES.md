# Changes

## LightGBM target scaling options (2026-11-13)

**What:** one global LightGBM, five ways to handle series of very different size, all trained,
calibrated and scored on the same windows and series:

| Method id | Target | Loss |
|---|---|---|
| `lgbm_level` (existing) | units | Tweedie |
| `lgbm_ratio` (existing) | units ÷ trailing 26-week mean | Tweedie |
| `lgbm_log1p` (new) | log(1 + units); back via expm1 × smearing factor | squared error |
| `lgbm_local_scaled` (new) | per-series standardised (`LocalStandardScaler`) | squared error |
| `lgbm_level_weighted` (new) | units, rows weighted by 52-week volume | Tweedie |

**Results (test WAPE, backtest mean in brackets):**

| Variant | WAPE all | WAPE sparse | Bias all | Pinball all |
|---|---:|---:|---:|---:|
| `lgbm_level` | **41.1%** (**46.4**) | **58.4%** (75.9) | −4.2% (−1.9) | **28.1%** (**31.5**) |
| `lgbm_ratio` | **41.1%** (47.0) | **58.4%** (**74.1**) | −2.5% (−0.8) | 28.3% (31.8) |
| `lgbm_log1p` | 43.7% (47.3) | 63.2% (74.7) | −11.2% (−9.6) | 29.3% (31.8) |
| `lgbm_local_scaled` | 41.7% (47.5) | 59.6% (76.5) | −1.6% (+2.1) | 28.5% (32.1) |
| `lgbm_level_weighted` | 41.7% (47.6) | 60.3% (80.7) | **+0.3%** (+3.0) | 28.6% (31.9) |

No scaling (`lgbm_level`) wins WAPE in all 4 backtests. log1p is worst and still biased low after
smearing. Standardising and volume weights cost 0.5–1 WAPE point but remove most of the
under-forecast. Headline results and the ensemble are unchanged.

**Code**
- `models_lgbm.py`: each variant is a frozen `Spec` (target, objective, weights, exogenous features)
  in `SPECS`. Shared `_fit` for forecasting and feature importance.
- `models_lgbm.validate` rejects Tweedie / Poisson / Gamma with a signed target (standard scaling,
  differencing) before training, with an explanation. LightGBM would otherwise stop with
  `at least one target label is negative`.
- log1p and local scaling are MLForecast `target_transforms`, so lags are built on the transformed
  scale and the recursive forecast inverts it correctly. log1p adds Duan smearing so forecasts are
  not biased low.
- Pipeline: new methods in `pipeline.LGBM_SCALING`, calibrated like the other LightGBM variants,
  shown in the side tables only. The ensemble candidates are unchanged.
- Page: new card "LightGBM: scale the target or not" under Results.

**Docs:** [docs/lightgbm-target-scaling.md](docs/lightgbm-target-scaling.md): the options, the
`LocalStandardScaler` + Tweedie pitfall, why lags must be transformed with the target, smearing,
and how weights slow small series down. README tradeoffs and pipeline sections link to it.

**Tests:** every variant forecasts every series on its own scale; the Tweedie / Poisson / Gamma +
signed-target guard; every registered `Spec` is valid; volume weights follow trailing volume and
average 1.

**Limitations:** one dataset (M5), one feature set, no hyperparameter search per variant. The
weighted variant may need more trees than the shared 400. Smearing uses in-sample residuals and
corrects slightly too little.
