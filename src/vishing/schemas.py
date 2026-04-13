"""Типизированные структуры данных проекта."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetItem:
    """Описание одного аудиофайла датасета."""

    path: Path
    dataset_root: Path
    relative_path: Path
    filename: str
    split: str
    label_dir: str
    label: int | None
    group_id: str


@dataclass(slots=True)
class FeatureBuildResult:
    """Результат построения датасета признаков."""

    output_csv: Path
    output_parquet: Path | None
    row_count: int
    columns: list[str]


@dataclass(slots=True)
class TrainingArtifacts:
    """Пути и метаданные обученной модели."""

    model_name: str
    artifact_dir: Path
    report_path: Path
    metrics_path: Path
    threshold_path: Path
    validation_predictions_path: Path
    metrics: dict[str, Any]
    threshold: float


@dataclass(slots=True)
class PredictionArtifacts:
    """Результат инференса по папке."""

    predictions_path: Path
    debug_predictions_path: Path
    row_count: int
