"""Формирование markdown- и json-отчётов."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vishing.constants import (
    ERROR_ANALYSIS_MD_NAME,
    FEATURE_OVERVIEW_MD_NAME,
    METRICS_SUMMARY_JSON_NAME,
    METRICS_SUMMARY_MD_NAME,
    models_dir,
    reports_dir,
)
from vishing.eval.error_analysis import write_error_analysis
from vishing.models.save_load import load_json, save_json
from vishing.utils.io import ensure_dir, write_text_file


def write_training_report(path: Path, title: str, lines: list[str]) -> None:
    """Сохранить человекочитаемый markdown-отчёт."""

    write_text_file(path, "\n".join([f"# {title}", "", *lines, ""]) + "\n")


def rebuild_metrics_summary(dataset_root: Path) -> tuple[Path, Path]:
    """Собрать общий summary по доступным метрикам моделей."""

    models_root = models_dir(dataset_root)
    reports_root = ensure_dir(reports_dir(dataset_root))
    summary: dict[str, dict] = {}
    for model_dir in sorted(path for path in models_root.glob("*") if path.is_dir()):
        metrics_path = model_dir / "metrics.json"
        threshold_path = model_dir / "threshold.json"
        if not metrics_path.exists():
            continue
        payload = {"metrics": load_json(metrics_path)}
        if threshold_path.exists():
            payload["threshold"] = load_json(threshold_path)
        summary[model_dir.name] = payload

    json_path = reports_root / METRICS_SUMMARY_JSON_NAME
    md_path = reports_root / METRICS_SUMMARY_MD_NAME
    save_json(json_path, summary)

    lines = ["# Сводка метрик", ""]
    if not summary:
        lines.append("Артефакты моделей пока не найдены.")
    else:
        for model_name, payload in summary.items():
            metrics = payload.get("metrics", {})
            threshold = payload.get("threshold", {}).get("threshold")
            lines.extend(
                [
                    f"## {model_name}",
                    "",
                    f"- accuracy: {metrics.get('accuracy')}",
                    f"- precision: {metrics.get('precision')}",
                    f"- recall: {metrics.get('recall')}",
                    f"- f1: {metrics.get('f1')}",
                    f"- roc_auc: {metrics.get('roc_auc')}",
                    f"- threshold: {threshold}",
                    "",
                ]
            )
    write_text_file(md_path, "\n".join(lines) + "\n")
    return json_path, md_path


def rebuild_error_analysis(dataset_root: Path) -> Path:
    """Построить единый markdown-анализ ошибок по лучшей доступной модели."""

    reports_root = ensure_dir(reports_dir(dataset_root))
    candidates = [
        dataset_root / "artifacts" / "models" / "ensemble" / "validation_predictions.csv",
        dataset_root / "artifacts" / "models" / "logreg" / "validation_predictions.csv",
        dataset_root / "artifacts" / "models" / "catboost" / "validation_predictions.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            output_path = reports_root / ERROR_ANALYSIS_MD_NAME
            write_error_analysis(frame, output_path, model_name=path.parent.name)
            return output_path

    output_path = reports_root / ERROR_ANALYSIS_MD_NAME
    write_text_file(output_path, "# Анализ ошибок\n\nValidation-предсказания пока не найдены.\n")
    return output_path


def feature_overview_path(dataset_root: Path) -> Path:
    """Вернуть путь к feature overview report."""

    return reports_dir(dataset_root) / FEATURE_OVERVIEW_MD_NAME
