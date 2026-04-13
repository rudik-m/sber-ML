"""Мелкие утилиты без побочных эффектов."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd

_GROUP_RE = re.compile(r"^(?P<prefix>.+?)_(?P<index>\d+)$")


def extract_group_id(name_or_filename: str) -> str:
    """Извлечь идентификатор серии из имени файла."""

    stem = Path(name_or_filename).stem.lower()
    match = _GROUP_RE.match(stem)
    if match:
        return match.group("prefix")
    return stem


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Безопасное деление."""

    if denominator == 0:
        return default
    return numerator / denominator


def semicolon_join(values: Iterable[str]) -> str:
    """Склеить строки через `; ` без пустых значений."""

    cleaned = [value for value in values if value]
    return "; ".join(cleaned)


def top_counts_to_text(items: dict[str, int], top_k: int = 3) -> str:
    """Преобразовать словарь счётчиков в компактную строку."""

    pairs = [(name, count) for name, count in items.items() if count > 0]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return semicolon_join(f"{name}:{count}" for name, count in pairs[:top_k])


def balanced_head(frame: pd.DataFrame, limit: int | None, label_col: str = "label") -> pd.DataFrame:
    """Ограничить датафрейм, сохранив обе метки, если это возможно."""

    if limit is None or len(frame) <= limit:
        return frame.copy()

    labeled = frame[frame[label_col].notna()]
    labels = list(dict.fromkeys(labeled[label_col].tolist()))
    if not labels:
        return frame.head(limit).copy()

    per_label = max(1, limit // max(len(labels), 1))
    chunks: list[pd.DataFrame] = []
    used_indexes: set[int] = set()
    for label in labels:
        chunk = labeled[labeled[label_col] == label].head(per_label)
        chunks.append(chunk)
        used_indexes.update(chunk.index.tolist())

    selected = pd.concat(chunks, axis=0) if chunks else frame.head(0)
    remaining_slots = max(limit - len(selected), 0)
    if remaining_slots > 0:
        remaining = frame.loc[~frame.index.isin(used_indexes)].head(remaining_slots)
        selected = pd.concat([selected, remaining], axis=0)

    return selected.head(limit).reset_index(drop=True)
