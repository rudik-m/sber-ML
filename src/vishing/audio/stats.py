"""Вычисление простых статистических признаков из аудио."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

from vishing.audio.vad import estimate_voice_activity
from vishing.utils.io import read_yaml_file


def load_audio_feature_config(path: Path) -> dict[str, Any]:
    """Загрузить конфиг аудиопризнаков."""

    return read_yaml_file(path)


def empty_audio_features() -> dict[str, float | str]:
    """Вернуть пустой набор аудиопризнаков."""

    return {
        "duration_sec": float("nan"),
        "rms_mean": float("nan"),
        "rms_std": float("nan"),
        "zero_crossing_rate_mean": float("nan"),
        "spectral_centroid_mean": float("nan"),
        "spectral_bandwidth_mean": float("nan"),
        "silence_ratio": float("nan"),
        "estimated_speech_ratio": float("nan"),
        "mean_pause_duration": float("nan"),
        "pause_count": float("nan"),
        "estimated_words_per_minute": float("nan"),
        "audio_feature_error": "",
    }


def extract_audio_features(path: Path, transcript_word_count: int, config: dict[str, Any]) -> dict[str, float | str]:
    """Вычислить лёгкие признаки из wav-файла."""

    features = empty_audio_features()
    try:
        sample_rate = int(config.get("sample_rate", 16_000))
        frame_length = int(config.get("frame_length", 1024))
        hop_length = int(config.get("hop_length", 512))
        signal, sr = librosa.load(path, sr=sample_rate, mono=True)
        if signal.size == 0:
            raise ValueError("пустой аудиосигнал")

        duration_sec = float(signal.size) / float(sr)
        rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]

        frame_duration_sec = hop_length / sr
        vad_stats = estimate_voice_activity(
            rms,
            frame_duration_sec,
            energy_quantile=float(config.get("energy_quantile", 0.2)),
            threshold_scale=float(config.get("energy_threshold_scale", 1.4)),
            min_rms_threshold=float(config.get("min_rms_threshold", 0.0005)),
            min_pause_duration_sec=float(config.get("min_pause_duration_sec", 0.3)),
        )

        features.update(
            {
                "duration_sec": round(duration_sec, 3),
                "rms_mean": round(float(np.mean(rms)), 6),
                "rms_std": round(float(np.std(rms)), 6),
                "zero_crossing_rate_mean": round(float(np.mean(zcr)), 6),
                "spectral_centroid_mean": round(float(np.mean(spectral_centroid)), 6),
                "spectral_bandwidth_mean": round(float(np.mean(spectral_bandwidth)), 6),
                "silence_ratio": vad_stats.silence_ratio,
                "estimated_speech_ratio": vad_stats.speech_ratio,
                "mean_pause_duration": vad_stats.mean_pause_duration,
                "pause_count": float(vad_stats.pause_count),
                "estimated_words_per_minute": round((transcript_word_count / duration_sec) * 60.0, 3)
                if duration_sec > 0
                else 0.0,
            }
        )
        return features
    except Exception as exc:
        features["audio_feature_error"] = f"{exc.__class__.__name__}: {exc}"
        return features
