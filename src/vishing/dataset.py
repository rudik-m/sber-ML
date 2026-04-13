"""Работа со структурой датасета и артефактами."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from vishing.constants import KNOWN_LABEL_DIRS, LABEL_TO_ID, SUPPORTED_SPLITS, transcripts_csv_path
from vishing.schemas import DatasetItem
from vishing.utils.misc import extract_group_id


def resolve_input_dir(input_dir: Path) -> Path:
    """Проверить и нормализовать входной каталог."""

    resolved = input_dir.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Входной каталог не найден: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Ожидался каталог, а не файл: {resolved}")
    return resolved


def resolve_dataset_root(input_dir: Path) -> Path:
    """Определить корень датасета `vishing` по переданному пути."""

    resolved = resolve_input_dir(input_dir)
    if resolved.name in SUPPORTED_SPLITS:
        return resolved.parent.resolve()
    if resolved.name in KNOWN_LABEL_DIRS and resolved.parent.name in SUPPORTED_SPLITS:
        return resolved.parent.parent.resolve()
    return resolved


def discover_dataset_items(input_dir: Path, limit: int | None = None) -> list[DatasetItem]:
    """Найти `wav`-файлы для корня датасета или его подкаталога."""

    target_dir = resolve_input_dir(input_dir)
    dataset_root = resolve_dataset_root(target_dir)

    audio_paths: list[Path] = []
    if target_dir == dataset_root and any((dataset_root / split).exists() for split in SUPPORTED_SPLITS):
        for split in SUPPORTED_SPLITS:
            split_dir = dataset_root / split
            if split_dir.exists():
                audio_paths.extend(sorted(path for path in split_dir.rglob("*.wav") if path.is_file()))
    else:
        audio_paths.extend(sorted(path for path in target_dir.rglob("*.wav") if path.is_file()))

    items: list[DatasetItem] = []
    for audio_path in audio_paths:
        try:
            relative_path = audio_path.relative_to(dataset_root)
        except ValueError:
            relative_path = Path(audio_path.name)

        parts = relative_path.parts
        split = parts[0] if parts and parts[0] in SUPPORTED_SPLITS else ""
        label_dir = next((part for part in parts if part in KNOWN_LABEL_DIRS), "")
        label = LABEL_TO_ID.get(label_dir)
        items.append(
            DatasetItem(
                path=audio_path,
                dataset_root=dataset_root,
                relative_path=relative_path,
                filename=audio_path.name,
                split=split,
                label_dir=label_dir,
                label=label,
                group_id=extract_group_id(audio_path.name),
            )
        )

    if limit is not None:
        return items[:limit]
    return items


def load_transcripts_frame(dataset_root: Path) -> pd.DataFrame:
    """Прочитать агрегированный CSV транскриптов, если он уже есть."""

    csv_path = transcripts_csv_path(dataset_root)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def count_audio_by_bucket(input_dir: Path) -> dict[str, int]:
    """Посчитать количество `wav` по сочетанию `split/label_dir`."""

    counter: Counter[str] = Counter()
    for item in discover_dataset_items(input_dir):
        bucket = "/".join(part for part in [item.split, item.label_dir] if part)
        counter[bucket or "root"] += 1
    return dict(sorted(counter.items()))
