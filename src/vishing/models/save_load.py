"""Сохранение и загрузка артефактов моделей."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from vishing.utils.io import ensure_dir, read_json_file, read_yaml_file, write_json_file, write_yaml_file


def save_joblib(obj: Any, path: Path) -> None:
    """Сохранить Python-объект через joblib."""

    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    """Загрузить Python-объект через joblib."""

    return joblib.load(path)


def save_model_config(path: Path, payload: dict[str, Any]) -> None:
    """Сохранить конфиг модели в YAML."""

    write_yaml_file(path, payload)


def load_model_config(path: Path) -> dict[str, Any]:
    """Загрузить конфиг модели из YAML."""

    return read_yaml_file(path)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Сохранить JSON-артефакт."""

    write_json_file(path, payload)


def load_json(path: Path) -> dict[str, Any]:
    """Загрузить JSON-артефакт."""

    return read_json_file(path)
