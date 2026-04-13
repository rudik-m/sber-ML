"""Файловые утилиты для этапа транскрибации."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

CSV_COLUMNS = [
    "filepath",
    "split",
    "label_dir",
    "filename",
    "transcript",
    "language",
    "duration_sec",
    "backend",
    "model_name",
    "error",
]
SUPPORTED_SPLITS = ("samples", "test")


@dataclass(frozen=True, slots=True)
class AudioSource:
    """Метаданные одного исходного аудиофайла."""

    path: Path
    relative_path: Path
    split: str
    label_dir: str
    filename: str


def resolve_input_dir(input_dir: Path) -> Path:
    """Нормализовать и проверить корень датасета."""

    resolved = input_dir.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {resolved}")
    return resolved


def resolve_output_dir(input_dir: Path, output_dir: Path | None) -> Path:
    """Проверить, что артефакты будут писаться внутрь ``input_dir/artifacts``."""

    default_output_dir = input_dir / "artifacts" / "transcripts"
    resolved = (output_dir or default_output_dir).expanduser().resolve()
    artifacts_root = (input_dir / "artifacts").resolve()
    if not resolved.is_relative_to(artifacts_root):
        raise ValueError(f"Output directory must stay inside {artifacts_root}")
    return resolved


def discover_audio_sources(input_dir: Path, limit: int | None = None) -> list[AudioSource]:
    """Собрать ``.wav``-файлы из поддерживаемых split-ов."""

    sources: list[AudioSource] = []
    for split in SUPPORTED_SPLITS:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue

        for audio_path in sorted(split_dir.rglob("*.wav")):
            if not audio_path.is_file():
                continue

            relative_path = audio_path.relative_to(input_dir)
            parts = relative_path.parts
            label_dir = parts[1] if len(parts) > 1 else ""
            sources.append(
                AudioSource(
                    path=audio_path,
                    relative_path=relative_path,
                    split=split,
                    label_dir=label_dir,
                    filename=audio_path.name,
                )
            )

    if limit is not None:
        return sources[:limit]
    return sources


def transcript_text_path(output_dir: Path, source: AudioSource) -> Path:
    """Вернуть путь для текстового транскрипта."""

    return output_dir / source.relative_path.with_suffix(".txt")


def transcript_json_path(output_dir: Path, source: AudioSource) -> Path:
    """Вернуть путь для JSON с сегментами."""

    return output_dir / source.relative_path.with_suffix(".json")


def ensure_parent_dir(path: Path) -> None:
    """Создать родительский каталог для файла."""

    path.parent.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path) -> str:
    """Прочитать UTF-8 текст из файла."""

    return path.read_text(encoding="utf-8").strip()


def write_text_file(path: Path, text: str) -> None:
    """Записать UTF-8 текст в файл."""

    ensure_parent_dir(path)
    path.write_text(text, encoding="utf-8")


def read_json_file(path: Path) -> dict[str, Any]:
    """Прочитать JSON из файла."""

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    """Записать JSON с форматированием."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def write_transcripts_csv(records: list[dict[str, Any]], csv_path: Path) -> None:
    """Сохранить агрегированный CSV по транскриптам."""

    ensure_parent_dir(csv_path)
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=CSV_COLUMNS)
    else:
        frame = frame.sort_values("filepath").reindex(columns=CSV_COLUMNS)
    frame.to_csv(csv_path, index=False)


def write_yaml_file(path: Path, payload: dict[str, Any]) -> None:
    """Сохранить YAML для воспроизводимости запуска."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, allow_unicode=True, sort_keys=False)
