"""Segment assignment from training history only. Pure, deterministic.

Order of precedence: new -> sparse -> hit -> middle.
* new:    fewer than `new_max_weeks` weeks since the first sale
* sparse: share of zero-sales weeks in the trailing 52 weeks >= threshold
* hit:    top `hit_top_share` of the remaining (established, regular) series by trailing-52w volume
* middle: everything else
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEGMENT_RULES, SegmentRules


def series_profile(train: pd.DataFrame, cutoff: int, lookback: int = 52) -> pd.DataFrame:
    """Per-series stats at the cutoff: age, trailing volume and zero share."""
    first = train.groupby("unique_id")["week"].min()
    recent = train[train["week"] > cutoff - lookback]
    prof = pd.DataFrame(
        {
            "age_weeks": cutoff - first + 1,
            "volume_52w": recent.groupby("unique_id")["y"].sum(),
            "zero_share_52w": (recent["y"] == 0).groupby(recent["unique_id"]).mean(),
        }
    )
    return prof.fillna({"volume_52w": 0.0, "zero_share_52w": 1.0})


def assign_segments(profile: pd.DataFrame, rules: SegmentRules = SEGMENT_RULES) -> pd.Series:
    seg = pd.Series("middle", index=profile.index, name="segment", dtype=object)
    is_new = profile["age_weeks"] < rules.new_max_weeks
    is_sparse = ~is_new & (profile["zero_share_52w"] >= rules.sparse_zero_share)
    regular = ~is_new & ~is_sparse
    if regular.any():
        threshold = np.quantile(profile.loc[regular, "volume_52w"], 1 - rules.hit_top_share)
        is_hit = regular & (profile["volume_52w"] >= threshold)
    else:
        is_hit = regular
    seg[is_hit] = "hit"
    seg[is_sparse] = "sparse"
    seg[is_new] = "new"
    return seg
