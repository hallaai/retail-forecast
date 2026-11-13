"""Chronos-Bolt zero-shot: no training, the pretrained model reads each series' history.

Univariate on purpose (no price / calendar covariates) so it is a clean zero-shot test.

Point forecast: Chronos-Bolt's own "mean" output is its median (chronos_bolt.py says so), and for
right-skewed, zero-heavy demand the median sits well below the mean. We therefore approximate the
mean as the average of the 9 native deciles (0.1..0.9) and keep the median in q50. The tails beyond
the 10th / 90th percentile are ignored, so this still slightly understates a right-skewed mean.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from . import config
from .forecast_frame import standardize

log = logging.getLogger(__name__)
METHOD = "chronos_bolt"
DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
_PIPELINE = None


def _pipeline():
    global _PIPELINE  # loaded once per process; the model is read-only after loading
    if _PIPELINE is None:
        from chronos import BaseChronosPipeline

        torch.set_num_threads(12)
        _PIPELINE = BaseChronosPipeline.from_pretrained(config.CHRONOS_MODEL, device_map="cpu", torch_dtype=torch.float32)
    return _PIPELINE


def warm_up() -> None:
    _pipeline()


def forecast(train: pd.DataFrame, cutoff: int, horizon: int) -> pd.DataFrame:
    ordered = train.sort_values(["unique_id", "week"])
    groups = ordered.groupby("unique_id", sort=True)["y"]
    ids = np.array(list(groups.groups.keys()))
    contexts = [torch.tensor(g.to_numpy(), dtype=torch.float32) for _, g in groups]
    pipe = _pipeline()
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(contexts), config.CHRONOS_BATCH):
            q, _median = pipe.predict_quantiles(contexts[i : i + config.CHRONOS_BATCH], prediction_length=horizon, quantile_levels=DECILES)
            chunks.append(q.numpy())
    q = np.concatenate(chunks)  # (n_series, horizon, 9)
    weeks = np.arange(cutoff + 1, cutoff + horizon + 1)
    frame = pd.DataFrame(
        {
            "unique_id": np.repeat(ids, horizon),
            "week": np.tile(weeks, len(ids)),
            "f": q.mean(axis=2).reshape(-1),
            "q10": q[:, :, 0].reshape(-1),
            "q50": q[:, :, 4].reshape(-1),
            "q90": q[:, :, 8].reshape(-1),
        }
    )
    return standardize(frame)
