"""Экспорт датасета признаков и краткий обзор."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vishing.utils.io import save_dataframe_csv, save_dataframe_parquet, write_text_file


def save_features_dataset(frame: pd.DataFrame, csv_path: Path, parquet_path: Path | None = None) -> Path | None:
    """Сохранить датасет признаков на диск."""

    save_dataframe_csv(frame, csv_path)
    if parquet_path is None:
        return None
    saved = save_dataframe_parquet(frame, parquet_path)
    return parquet_path if saved else None


def write_feature_overview(frame: pd.DataFrame, path: Path) -> None:
    """Сформировать краткий markdown-обзор по признакам."""

    label_stats = frame["label_dir"].fillna("unknown").value_counts().to_dict() if "label_dir" in frame else {}
    avg_words = round(float(frame["text_word_count"].mean()), 2) if "text_word_count" in frame and not frame.empty else 0.0
    avg_duration = round(float(frame["duration_sec"].mean()), 2) if "duration_sec" in frame and not frame.empty else 0.0
    trigger_mean = (
        round(float(frame["trigger_total_count"].mean()), 2) if "trigger_total_count" in frame and not frame.empty else 0.0
    )

    lines = [
        "# Обзор признаков",
        "",
        f"- Количество строк: {len(frame)}",
        f"- Среднее число слов в транскрипте: {avg_words}",
        f"- Средняя длительность аудио, сек: {avg_duration}",
        f"- Среднее число триггеров: {trigger_mean}",
        f"- Распределение по классам: {label_stats}",
    ]
    write_text_file(path, "\n".join(lines) + "\n")
