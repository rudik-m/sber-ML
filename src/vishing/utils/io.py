"""Утилиты чтения и записи файлов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: Path) -> Path:
    """Создать каталог, если его ещё нет."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> None:
    """Создать родительский каталог для файла."""

    path.parent.mkdir(parents=True, exist_ok=True)


def read_yaml_file(path: Path) -> dict[str, Any]:
    """Прочитать YAML-файл."""

    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Ожидался YAML-объект верхнего уровня: {path}")
    return payload


def write_yaml_file(path: Path, payload: dict[str, Any]) -> None:
    """Сохранить YAML-файл."""

    ensure_parent(path)
    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, allow_unicode=True, sort_keys=False)


def read_json_file(path: Path) -> dict[str, Any]:
    """Прочитать JSON-файл."""

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    """Сохранить JSON-файл."""

    ensure_parent(path)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def write_text_file(path: Path, text: str) -> None:
    """Сохранить текстовый файл."""

    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def save_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    """Сохранить DataFrame в CSV."""

    ensure_parent(path)
    frame.to_csv(path, index=False)


def save_dataframe_parquet(frame: pd.DataFrame, path: Path) -> bool:
    """Попытаться сохранить DataFrame в Parquet."""

    ensure_parent(path)
    try:
        frame.to_parquet(path, index=False)
    except Exception:
        return False
    return True
