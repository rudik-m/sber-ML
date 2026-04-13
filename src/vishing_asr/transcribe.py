"""Оркестрация этапа транскрибации."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re
import shutil
from pathlib import Path
from typing import Any, Protocol

from rich.console import Console
from tqdm import tqdm

from vishing_asr.audio import load_audio_mono_16k, safe_probe_duration
from vishing_asr.io_utils import (
    AudioSource,
    discover_audio_sources,
    read_json_file,
    read_text_file,
    transcript_json_path,
    transcript_text_path,
    write_json_file,
    write_text_file,
    write_transcripts_csv,
    write_yaml_file,
)


@dataclass(frozen=True, slots=True)
class TranscribeConfig:
    """Конфиг одного запуска транскрибации по набору файлов."""

    input_dir: Path
    output_dir: Path
    backend: str
    model_name: str
    device: str
    compute_type: str
    language: str | None
    overwrite: bool
    limit: int | None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Нормализованный результат ASR для одного файла."""

    transcript: str
    language: str
    segments: list[dict[str, Any]]


class ASRBackend(Protocol):
    """Минимальный контракт backend-а для дальнейшего расширения."""

    def transcribe(self, audio: Any, *, language: str | None) -> TranscriptionResult:
        """Расшифровать нормализованный аудиомассив."""


class FasterWhisperBackend:
    """Реализация backend-а на faster-whisper."""

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio: Any, *, language: str | None) -> TranscriptionResult:
        segments, info = self._model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
        )

        segment_payloads: list[dict[str, Any]] = []
        transcript_parts: list[str] = []
        for segment in segments:
            raw_text = segment.text or ""
            clean_text = normalize_whitespace(raw_text)
            if clean_text:
                transcript_parts.append(clean_text)

            segment_payloads.append(
                {
                    "id": int(segment.id),
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "text": clean_text,
                    "raw_text": raw_text,
                    "avg_logprob": maybe_round(getattr(segment, "avg_logprob", None)),
                    "compression_ratio": maybe_round(getattr(segment, "compression_ratio", None)),
                    "no_speech_prob": maybe_round(getattr(segment, "no_speech_prob", None)),
                }
            )

        detected_language = getattr(info, "language", None) or language or ""
        return TranscriptionResult(
            transcript=normalize_whitespace(" ".join(transcript_parts)),
            language=detected_language,
            segments=segment_payloads,
        )


def normalize_whitespace(text: str) -> str:
    """Схлопнуть повторные пробелы и переводы строк."""

    return re.sub(r"\s+", " ", text).strip()


def maybe_round(value: float | None) -> float | None:
    """Округлить опциональное число для стабильного вывода."""

    if value is None:
        return None
    return round(float(value), 6)


def normalize_language(language: str | None) -> str | None:
    """Преобразовать значение языка из CLI в формат backend-а."""

    if language is None:
        return None
    normalized = language.strip().lower()
    if not normalized or normalized == "auto":
        return None
    return normalized


def resolve_device(device: str) -> str:
    """Преобразовать ``auto`` в конкретное устройство."""

    if device != "auto":
        return device
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def resolve_compute_type(compute_type: str, resolved_device: str) -> str:
    """Преобразовать ``auto`` в подходящий compute type."""

    if compute_type != "auto":
        return compute_type
    return "float16" if resolved_device == "cuda" else "int8"


def build_backend(config: TranscribeConfig) -> ASRBackend:
    """Создать экземпляр выбранного backend-а."""

    if config.backend == "faster-whisper":
        return FasterWhisperBackend(
            model_name=config.model_name,
            device=config.device,
            compute_type=config.compute_type,
        )
    raise ValueError(f"Unsupported backend: {config.backend}")


def transcribe_dataset(config: TranscribeConfig, console: Console) -> list[dict[str, Any]]:
    """Запустить транскрибацию по нужному набору файлов."""

    sources = discover_audio_sources(config.input_dir, limit=config.limit)
    if not sources:
        raise FileNotFoundError(f"No .wav files found under {config.input_dir}")

    backend = build_backend(config)
    csv_path = config.output_dir / "transcripts.csv"

    records: list[dict[str, Any]] = []
    error_count = 0
    for source in tqdm(sources, desc="Transcribing", unit="file"):
        record = process_source(
            source=source,
            config=config,
            backend=backend,
            console=console,
        )
        if record["error"]:
            error_count += 1
        records.append(record)
        write_transcripts_csv(records, csv_path)

    write_yaml_file(
        config.output_dir / "run_config.yaml",
        {
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
            "input_dir": str(config.input_dir),
            "output_dir": str(config.output_dir),
            "backend": config.backend,
            "model_name": config.model_name,
            "device": config.device,
            "compute_type": config.compute_type,
            "language": config.language or "auto",
            "overwrite": config.overwrite,
            "limit": config.limit,
            "file_count": len(sources),
            "error_count": error_count,
        },
    )
    return records


def process_source(
    source: AudioSource,
    config: TranscribeConfig,
    backend: ASRBackend,
    console: Console,
) -> dict[str, Any]:
    """Транскрибировать один файл и сохранить текст/JSON артефакты."""

    text_path = transcript_text_path(config.output_dir, source)
    json_path = transcript_json_path(config.output_dir, source)

    if not config.overwrite and text_path.exists():
        cached_record = try_load_cached_record(source, config, text_path, json_path)
        if cached_record is not None:
            return cached_record

    duration_sec: float | None = None
    try:
        audio = load_audio_mono_16k(source.path)
        duration_sec = round(audio.duration_sec, 3)
        result = backend.transcribe(audio.samples, language=config.language)

        write_text_file(text_path, result.transcript)
        write_json_file(
            json_path,
            {
                "filepath": source.relative_path.as_posix(),
                "backend": config.backend,
                "model_name": config.model_name,
                "language": result.language,
                "duration_sec": duration_sec,
                "audio": {
                    "original_sample_rate": audio.original_sample_rate,
                    "target_sample_rate": audio.sample_rate,
                    "channels": audio.channels,
                },
                "segments": result.segments,
                "error": "",
            },
        )

        return build_record(
            source=source,
            transcript=result.transcript,
            language=result.language,
            duration_sec=duration_sec,
            backend=config.backend,
            model_name=config.model_name,
            error="",
        )
    except Exception as exc:
        error_message = format_error(exc)
        if duration_sec is None:
            probed_duration = safe_probe_duration(source.path)
            duration_sec = round(probed_duration, 3) if probed_duration is not None else None

        write_text_file(text_path, "")
        write_json_file(
            json_path,
            {
                "filepath": source.relative_path.as_posix(),
                "backend": config.backend,
                "model_name": config.model_name,
                "language": config.language or "",
                "duration_sec": duration_sec,
                "segments": [],
                "error": error_message,
            },
        )
        console.log(f"[red]Failed[/red] {source.relative_path.as_posix()}: {error_message}")
        return build_record(
            source=source,
            transcript="",
            language=config.language or "",
            duration_sec=duration_sec,
            backend=config.backend,
            model_name=config.model_name,
            error=error_message,
        )


def try_load_cached_record(
    source: AudioSource,
    config: TranscribeConfig,
    text_path: Path,
    json_path: Path,
) -> dict[str, Any] | None:
    """Переиспользовать готовые артефакты, если `overwrite` отключён."""

    try:
        transcript = read_text_file(text_path)
    except Exception:
        return None

    payload: dict[str, Any] = {}
    if json_path.exists():
        try:
            payload = read_json_file(json_path)
        except Exception:
            payload = {}

    duration_sec = payload.get("duration_sec")
    if duration_sec is None:
        probed_duration = safe_probe_duration(source.path)
        duration_sec = round(probed_duration, 3) if probed_duration is not None else None

    return build_record(
        source=source,
        transcript=transcript,
        language=str(payload.get("language") or config.language or ""),
        duration_sec=duration_sec,
        backend=str(payload.get("backend") or config.backend),
        model_name=str(payload.get("model_name") or config.model_name),
        error=str(payload.get("error") or ""),
    )


def build_record(
    source: AudioSource,
    transcript: str,
    language: str,
    duration_sec: float | None,
    backend: str,
    model_name: str,
    error: str,
) -> dict[str, Any]:
    """Собрать одну строку для итогового CSV."""

    return {
        "filepath": source.relative_path.as_posix(),
        "split": source.split,
        "label_dir": source.label_dir,
        "filename": source.filename,
        "transcript": transcript,
        "language": language,
        "duration_sec": duration_sec,
        "backend": backend,
        "model_name": model_name,
        "error": error,
    }


def format_error(exc: Exception) -> str:
    """Сформатировать компактную строку ошибки для логов и CSV."""

    return normalize_whitespace(f"{exc.__class__.__name__}: {exc}")
