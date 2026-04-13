"""Сборка единого датасета handcrafted-признаков."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from vishing.audio.stats import extract_audio_features, load_audio_feature_config
from vishing.constants import (
    CONFIGS_DIR,
    FEATURES_CSV_NAME,
    FEATURES_PARQUET_NAME,
    FEATURE_OVERVIEW_MD_NAME,
    features_dir,
    reports_dir,
)
from vishing.dataset import discover_dataset_items, load_transcripts_frame, resolve_dataset_root
from vishing.features.export import save_features_dataset, write_feature_overview
from vishing.schemas import FeatureBuildResult
from vishing.text.meta_features import extract_text_meta_features, load_text_feature_config
from vishing.text.normalize import normalize_text
from vishing.text.patterns import extract_pattern_features, load_pattern_catalog
from vishing.text.triggers import extract_trigger_features, load_trigger_catalog
from vishing.utils.io import ensure_dir
from vishing.utils.logging import get_console
from vishing_asr.io_utils import resolve_output_dir as resolve_asr_output_dir
from vishing_asr.transcribe import (
    TranscribeConfig as AsrTranscribeConfig,
    normalize_language,
    resolve_compute_type,
    resolve_device,
    transcribe_dataset,
)


def ensure_transcripts_for_items(
    items_relative_paths: list[str],
    dataset_root: Path,
    *,
    run_transcribe_missing: bool,
    limit_for_transcribe: int | None,
) -> pd.DataFrame:
    """Убедиться, что для нужных файлов уже есть агрегированный CSV транскриптов."""

    console = get_console()
    transcripts_frame = load_transcripts_frame(dataset_root)
    available_paths = set(transcripts_frame["filepath"].tolist()) if not transcripts_frame.empty else set()
    missing = [path for path in items_relative_paths if path not in available_paths]
    if not missing:
        return transcripts_frame

    if not run_transcribe_missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            "Не хватает транскриптов. Сначала запустите `uv run vishing transcribe --input-dir ./vishing` "
            f"или включите автодозапуск. Примеры отсутствующих файлов: {preview}"
        )

    console.log("[yellow]Не хватает транскриптов, запускаю существующий ASR-этап.[/yellow]")
    resolved_output_dir = resolve_asr_output_dir(dataset_root, None)
    resolved_device = resolve_device("auto")
    resolved_compute_type = resolve_compute_type("auto", resolved_device)
    config = AsrTranscribeConfig(
        input_dir=dataset_root,
        output_dir=resolved_output_dir,
        backend="faster-whisper",
        model_name="base",
        device=resolved_device,
        compute_type=resolved_compute_type,
        language=normalize_language("ru"),
        overwrite=False,
        limit=limit_for_transcribe,
    )
    transcribe_dataset(config, console=console)
    transcripts_frame = load_transcripts_frame(dataset_root)
    available_paths = set(transcripts_frame["filepath"].tolist()) if not transcripts_frame.empty else set()
    missing_after = [path for path in items_relative_paths if path not in available_paths]
    if missing_after:
        preview = ", ".join(missing_after[:5])
        raise FileNotFoundError(f"После запуска ASR всё ещё нет транскриптов для: {preview}")
    return transcripts_frame


def build_feature_frame(
    input_dir: Path,
    *,
    limit: int | None = None,
    run_transcribe_missing: bool = True,
) -> pd.DataFrame:
    """Построить датафрейм с признаками без сохранения на диск."""

    items = discover_dataset_items(input_dir, limit=limit)
    if not items:
        raise FileNotFoundError(f"Не найдено ни одного wav-файла в {input_dir}")

    dataset_root = resolve_dataset_root(input_dir)
    items_relative_paths = [item.relative_path.as_posix() for item in items]
    limit_for_transcribe = limit if Path(input_dir).resolve() == dataset_root.resolve() else None
    transcripts_frame = ensure_transcripts_for_items(
        items_relative_paths,
        dataset_root,
        run_transcribe_missing=run_transcribe_missing,
        limit_for_transcribe=limit_for_transcribe,
    )
    transcript_lookup = transcripts_frame.set_index("filepath").to_dict(orient="index")

    trigger_catalog = load_trigger_catalog(CONFIGS_DIR / "triggers.yaml")
    pattern_catalog = load_pattern_catalog(CONFIGS_DIR / "patterns.yaml")
    text_feature_config = load_text_feature_config(CONFIGS_DIR / "text_features.yaml")
    audio_feature_config = load_audio_feature_config(CONFIGS_DIR / "audio_features.yaml")

    rows: list[dict[str, Any]] = []
    for item in tqdm(items, desc="Сборка признаков", unit="file"):
        transcript_row = transcript_lookup.get(item.relative_path.as_posix())
        if transcript_row is None:
            raise KeyError(f"В transcripts.csv нет строки для {item.relative_path.as_posix()}")

        transcript_raw = str(transcript_row.get("transcript", "") or "")
        transcript_norm = normalize_text(transcript_raw)

        trigger_summary = extract_trigger_features(transcript_norm, trigger_catalog)
        pattern_summary = extract_pattern_features(transcript_norm, pattern_catalog)
        text_features = extract_text_meta_features(transcript_norm, text_feature_config)
        audio_features = extract_audio_features(
            item.path,
            transcript_word_count=int(text_features["text_word_count"]),
            config=audio_feature_config,
        )

        row: dict[str, Any] = {
            "filepath": item.relative_path.as_posix(),
            "filename": item.filename,
            "split": item.split,
            "label_dir": item.label_dir,
            "label": item.label,
            "group_id": item.group_id,
            "transcript_raw": transcript_raw,
            "transcript_norm": transcript_norm,
            "language": transcript_row.get("language", ""),
            "transcript_error": transcript_row.get("error", ""),
        }
        row.update(trigger_summary.features)
        row.update(pattern_summary.features)
        row.update(text_features)
        row.update(audio_features)
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("filepath").reset_index(drop=True)
    return frame


def build_and_save_features(
    input_dir: Path,
    *,
    output_csv: Path | None = None,
    limit: int | None = None,
    run_transcribe_missing: bool = True,
) -> FeatureBuildResult:
    """Построить и сохранить `features.csv`."""

    dataset_root = resolve_dataset_root(input_dir)
    frame = build_feature_frame(input_dir, limit=limit, run_transcribe_missing=run_transcribe_missing)

    default_features_dir = ensure_dir(features_dir(dataset_root))
    csv_path = output_csv or (default_features_dir / FEATURES_CSV_NAME)
    parquet_path = default_features_dir / FEATURES_PARQUET_NAME
    saved_parquet = save_features_dataset(frame, csv_path=csv_path, parquet_path=parquet_path)

    overview_path = ensure_dir(reports_dir(dataset_root)) / FEATURE_OVERVIEW_MD_NAME
    write_feature_overview(frame, overview_path)

    return FeatureBuildResult(
        output_csv=csv_path,
        output_parquet=saved_parquet,
        row_count=len(frame),
        columns=frame.columns.tolist(),
    )
