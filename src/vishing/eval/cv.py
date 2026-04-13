"""Тонкая обёртка над group-based holdout."""

from __future__ import annotations

import pandas as pd

from vishing.features.group_split import GroupSplit, make_group_holdout_split


def build_validation_split(frame: pd.DataFrame, config: dict) -> GroupSplit:
    """Построить одно честное holdout-разбиение по группам."""

    training_cfg = config.get("training", {})
    return make_group_holdout_split(
        frame,
        validation_size=float(training_cfg.get("validation_size", 0.25)),
        random_state=int(training_cfg.get("random_state", 42)),
        max_attempts=int(training_cfg.get("max_split_attempts", 100)),
    )
