"""Константы и пути проекта."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"

DEFAULT_DATASET_DIR = Path("./vishing")
SUPPORTED_SPLITS = ("samples", "test")
KNOWN_LABEL_DIRS = ("Fraud", "NotFraud")

FRAUD_LABEL = 0
NOT_FRAUD_LABEL = 1
LABEL_TO_ID = {"Fraud": FRAUD_LABEL, "NotFraud": NOT_FRAUD_LABEL}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

ARTIFACTS_DIRNAME = "artifacts"
TRANSCRIPTS_DIRNAME = "transcripts"
FEATURES_DIRNAME = "features"
MODELS_DIRNAME = "models"
PREDICTIONS_DIRNAME = "predictions"
REPORTS_DIRNAME = "reports"
LOGS_DIRNAME = "logs"

TRANSCRIPTS_CSV_NAME = "transcripts.csv"
FEATURES_CSV_NAME = "features.csv"
FEATURES_PARQUET_NAME = "features.parquet"
PREDICTION_FEATURES_CSV_NAME = "prediction_features.csv"
PREDICTIONS_CSV_NAME = "predictions.csv"
PREDICTIONS_DEBUG_CSV_NAME = "predictions_debug.csv"
METRICS_SUMMARY_JSON_NAME = "metrics_summary.json"
METRICS_SUMMARY_MD_NAME = "metrics_summary.md"
FEATURE_OVERVIEW_MD_NAME = "feature_overview.md"
ERROR_ANALYSIS_MD_NAME = "error_analysis.md"


def artifacts_root(dataset_root: Path) -> Path:
    """Вернуть корень артефактов."""

    return dataset_root / ARTIFACTS_DIRNAME


def transcripts_dir(dataset_root: Path) -> Path:
    """Вернуть каталог с транскриптами."""

    return artifacts_root(dataset_root) / TRANSCRIPTS_DIRNAME


def transcripts_csv_path(dataset_root: Path) -> Path:
    """Вернуть путь к агрегированному CSV транскриптов."""

    return transcripts_dir(dataset_root) / TRANSCRIPTS_CSV_NAME


def features_dir(dataset_root: Path) -> Path:
    """Вернуть каталог с датасетом признаков."""

    return artifacts_root(dataset_root) / FEATURES_DIRNAME


def features_csv_path(dataset_root: Path) -> Path:
    """Вернуть путь к основному CSV признаков."""

    return features_dir(dataset_root) / FEATURES_CSV_NAME


def models_dir(dataset_root: Path) -> Path:
    """Вернуть каталог с моделями."""

    return artifacts_root(dataset_root) / MODELS_DIRNAME


def predictions_dir(dataset_root: Path) -> Path:
    """Вернуть каталог с предсказаниями."""

    return artifacts_root(dataset_root) / PREDICTIONS_DIRNAME


def reports_dir(dataset_root: Path) -> Path:
    """Вернуть каталог с отчётами."""

    return artifacts_root(dataset_root) / REPORTS_DIRNAME


def logs_dir(dataset_root: Path) -> Path:
    """Вернуть каталог с логами."""

    return artifacts_root(dataset_root) / LOGS_DIRNAME
