"""Утилиты загрузки аудио для этапа транскрибации."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faster_whisper.audio import decode_audio
import soundfile as sf

TARGET_SAMPLE_RATE = 16_000


@dataclass(slots=True)
class AudioData:
    """Нормализованный аудиосигнал для ASR backend."""

    samples: Any
    sample_rate: int
    duration_sec: float
    original_sample_rate: int
    channels: int


def probe_audio(path: Path) -> tuple[int, int, float]:
    """Вернуть исходную частоту, число каналов и длительность в секундах."""

    info = sf.info(str(path))
    sample_rate = int(info.samplerate)
    channels = int(info.channels)
    duration_sec = float(info.frames) / float(sample_rate) if sample_rate else 0.0
    return sample_rate, channels, duration_sec


def safe_probe_duration(path: Path) -> float | None:
    """Вернуть длительность, если метаданные читаются корректно."""

    try:
        _, _, duration_sec = probe_audio(path)
    except Exception:
        return None
    return duration_sec


def load_audio_mono_16k(path: Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> AudioData:
    """Загрузить аудио и привести его к mono 16 kHz через faster-whisper."""

    original_sample_rate, channels, duration_sec = probe_audio(path)
    samples = decode_audio(str(path), sampling_rate=target_sample_rate)
    normalized_duration = float(len(samples)) / float(target_sample_rate) if len(samples) else duration_sec

    return AudioData(
        samples=samples,
        sample_rate=target_sample_rate,
        duration_sec=normalized_duration,
        original_sample_rate=original_sample_rate,
        channels=channels,
    )
