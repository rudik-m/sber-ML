"""Единый CLI для всего vishing-пайплайна."""

from __future__ import annotations

from pathlib import Path

import click

from vishing.constants import DEFAULT_DATASET_DIR, features_csv_path
from vishing.dataset import count_audio_by_bucket, discover_dataset_items, load_transcripts_frame, resolve_dataset_root, resolve_input_dir
from vishing.eval.reports import rebuild_error_analysis, rebuild_metrics_summary
from vishing.features.build import build_and_save_features, build_feature_frame
from vishing.models.ensemble import build_ensemble
from vishing.models.predict import predict_model, save_predictions
from vishing.models.train_catboost import train_catboost_model
from vishing.models.train_logreg import train_logreg_model
from vishing.utils.logging import configure_logging, get_console
from vishing_asr.io_utils import resolve_output_dir as resolve_asr_output_dir
from vishing_asr.transcribe import (
    TranscribeConfig,
    normalize_language,
    resolve_compute_type,
    resolve_device,
    transcribe_dataset,
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """CLI для офлайн-пайплайна распознавания мошеннических разговоров."""

    configure_logging()


@main.command()
@click.option("--input-dir", type=click.Path(path_type=Path, file_okay=False), default=DEFAULT_DATASET_DIR, show_default=True)
def inspect(input_dir: Path) -> None:
    """Показать, что найдено в данных и артефактах."""

    console = get_console()
    resolved_input = resolve_input_dir(input_dir)
    dataset_root = resolve_dataset_root(resolved_input)
    items = discover_dataset_items(resolved_input)
    transcripts_frame = load_transcripts_frame(dataset_root)
    feature_csv = features_csv_path(dataset_root)
    models_root = dataset_root / "artifacts" / "models"
    model_dirs = sorted(path.name for path in models_root.glob("*") if path.is_dir()) if models_root.exists() else []

    console.print("[bold]Сводка по проекту[/bold]")
    console.print(f"input_dir={resolved_input}")
    console.print(f"dataset_root={dataset_root}")
    console.print(f"wav_files={len(items)}")
    console.print(f"по каталогам={count_audio_by_bucket(resolved_input)}")
    console.print(f"transcripts_rows={len(transcripts_frame)}")
    console.print(f"features_csv_exists={feature_csv.exists()}")
    console.print(f"models={model_dirs}")
    for item in items[:5]:
        console.print(f"- пример: {item.relative_path.as_posix()} -> group_id={item.group_id}")


@main.command()
@click.option("--input-dir", type=click.Path(path_type=Path, file_okay=False), default=DEFAULT_DATASET_DIR, show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option("--backend", type=click.Choice(["faster-whisper"], case_sensitive=False), default="faster-whisper", show_default=True)
@click.option("--model-size", type=click.Choice(["tiny", "base", "small", "medium"], case_sensitive=False), default="base", show_default=True)
@click.option("--device", type=click.Choice(["cpu", "cuda", "auto"], case_sensitive=False), default="auto", show_default=True)
@click.option("--compute-type", type=click.Choice(["int8", "float16", "auto"], case_sensitive=False), default="auto", show_default=True)
@click.option("--language", default="ru", show_default=True)
@click.option("--overwrite", is_flag=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
def transcribe(
    input_dir: Path,
    output_dir: Path | None,
    backend: str,
    model_size: str,
    device: str,
    compute_type: str,
    language: str,
    overwrite: bool,
    limit: int | None,
) -> None:
    """Обёртка над уже реализованным ASR-этапом."""

    console = get_console()
    resolved_input_dir = resolve_input_dir(input_dir)
    resolved_output_dir = resolve_asr_output_dir(resolved_input_dir, output_dir)
    resolved_device = resolve_device(device.lower())
    resolved_compute = resolve_compute_type(compute_type.lower(), resolved_device)
    config = TranscribeConfig(
        input_dir=resolved_input_dir,
        output_dir=resolved_output_dir,
        backend=backend.lower(),
        model_name=model_size.lower(),
        device=resolved_device,
        compute_type=resolved_compute,
        language=normalize_language(language),
        overwrite=overwrite,
        limit=limit,
    )
    records = transcribe_dataset(config, console=console)
    error_count = sum(1 for record in records if record["error"])
    console.print(
        f"[green]Готово[/green]: обработано {len(records)} файлов, ошибок: {error_count}, CSV: {resolved_output_dir / 'transcripts.csv'}"
    )


@main.command("build-features")
@click.option("--input-dir", type=click.Path(path_type=Path, file_okay=False), default=DEFAULT_DATASET_DIR, show_default=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option("--run-transcribe-missing/--no-run-transcribe-missing", default=True, show_default=True)
def build_features(input_dir: Path, limit: int | None, run_transcribe_missing: bool) -> None:
    """Построить `features.csv` из transcript- и audio-артефактов."""

    console = get_console()
    result = build_and_save_features(
        input_dir=input_dir,
        limit=limit,
        run_transcribe_missing=run_transcribe_missing,
    )
    console.print(
        f"[green]Готово[/green]: features.csv={result.output_csv}, строк={result.row_count}, parquet={result.output_parquet}"
    )


@main.command("train-logreg")
@click.option("--features", "features_path", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--limit", type=click.IntRange(min=4), default=None)
def train_logreg(features_path: Path | None, limit: int | None) -> None:
    """Обучить baseline LogisticRegression."""

    console = get_console()
    actual_features_path = features_path or features_csv_path(resolve_dataset_root(DEFAULT_DATASET_DIR))
    result = train_logreg_model(actual_features_path, limit=limit)
    rebuild_metrics_summary(actual_features_path.resolve().parents[2])
    rebuild_error_analysis(actual_features_path.resolve().parents[2])
    console.print(f"[green]Готово[/green]: модель={result.model_name}, отчёт={result.report_path}")


@main.command("train-catboost")
@click.option("--features", "features_path", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--limit", type=click.IntRange(min=4), default=None)
def train_catboost(features_path: Path | None, limit: int | None) -> None:
    """Обучить основную модель CatBoost."""

    console = get_console()
    actual_features_path = features_path or features_csv_path(resolve_dataset_root(DEFAULT_DATASET_DIR))
    result = train_catboost_model(actual_features_path, limit=limit)
    rebuild_metrics_summary(actual_features_path.resolve().parents[2])
    rebuild_error_analysis(actual_features_path.resolve().parents[2])
    console.print(f"[green]Готово[/green]: модель={result.model_name}, отчёт={result.report_path}")


@main.command("train-all")
@click.option("--features", "features_path", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--limit", type=click.IntRange(min=4), default=None)
def train_all(features_path: Path | None, limit: int | None) -> None:
    """Обучить LogisticRegression, CatBoost и ансамбль."""

    console = get_console()
    actual_features_path = features_path or features_csv_path(resolve_dataset_root(DEFAULT_DATASET_DIR))
    logreg_result = train_logreg_model(actual_features_path, limit=limit)
    catboost_result = train_catboost_model(actual_features_path, limit=limit)
    ensemble_result = build_ensemble(actual_features_path.resolve().parents[2])
    rebuild_metrics_summary(actual_features_path.resolve().parents[2])
    rebuild_error_analysis(actual_features_path.resolve().parents[2])
    console.print(
        "[green]Готово[/green]: "
        f"logreg={logreg_result.artifact_dir}, catboost={catboost_result.artifact_dir}, ensemble={ensemble_result.artifact_dir}"
    )


@main.command()
@click.option("--input-dir", type=click.Path(path_type=Path, file_okay=False), default=DEFAULT_DATASET_DIR / "test", show_default=True)
@click.option("--model", "model_name", type=click.Choice(["logreg", "catboost", "ensemble"], case_sensitive=False), default="ensemble", show_default=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option("--run-transcribe-missing/--no-run-transcribe-missing", default=True, show_default=True)
def predict(input_dir: Path, model_name: str, limit: int | None, run_transcribe_missing: bool) -> None:
    """Построить признаки для папки и выдать `predictions.csv`."""

    console = get_console()
    dataset_root = resolve_dataset_root(input_dir)
    feature_frame = build_feature_frame(
        input_dir=input_dir,
        limit=limit,
        run_transcribe_missing=run_transcribe_missing,
    )
    fraud_scores, threshold = predict_model(feature_frame, dataset_root=dataset_root, model_name=model_name)
    artifacts = save_predictions(feature_frame, fraud_scores, threshold, dataset_root=dataset_root, model_name=model_name)
    console.print(
        f"[green]Готово[/green]: predictions={artifacts.predictions_path}, debug={artifacts.debug_predictions_path}, строк={artifacts.row_count}"
    )


@main.command()
@click.option("--input-dir", type=click.Path(path_type=Path, file_okay=False), default=DEFAULT_DATASET_DIR, show_default=True)
def report(input_dir: Path) -> None:
    """Собрать краткий markdown-отчёт по текущим артефактам."""

    console = get_console()
    dataset_root = resolve_dataset_root(input_dir)
    metrics_json, metrics_md = rebuild_metrics_summary(dataset_root)
    error_md = rebuild_error_analysis(dataset_root)
    console.print(
        f"[green]Готово[/green]: metrics_json={metrics_json}, metrics_md={metrics_md}, error_analysis={error_md}"
    )


if __name__ == "__main__":
    main()
