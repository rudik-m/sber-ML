"""Извлечение `group_id` и честное group-based разбиение."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from vishing.utils.misc import extract_group_id


@dataclass(frozen=True, slots=True)
class GroupSplit:
    """Индексы train/validation для одного holdout-разбиения."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    random_state: int


def make_group_holdout_split(
    frame: pd.DataFrame,
    *,
    label_col: str = "label",
    group_col: str = "group_id",
    validation_size: float = 0.25,
    random_state: int = 42,
    max_attempts: int = 100,
) -> GroupSplit:
    """Подобрать group-based holdout без вырождения по классам."""

    if frame.empty:
        raise ValueError("Пустой датафрейм нельзя разделить на train/validation.")

    groups = frame[group_col].fillna(frame["filename"]).astype(str)
    labels = frame[label_col].astype(int)
    for offset in range(max_attempts):
        seed = random_state + offset
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(frame, labels, groups))
        train_labels = set(labels.iloc[train_idx].tolist())
        val_labels = set(labels.iloc[val_idx].tolist())
        if len(train_labels) > 1 and len(val_labels) > 1:
            return GroupSplit(train_idx=train_idx, val_idx=val_idx, random_state=seed)

    raise RuntimeError("Не удалось подобрать group-based split с обеими метками в train и validation.")
