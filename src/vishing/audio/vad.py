"""Простой energy-based VAD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class VadStats:
    """Сводка по оценке речевой активности."""

    speech_ratio: float
    silence_ratio: float
    pause_count: int
    mean_pause_duration: float
    threshold: float


def estimate_voice_activity(
    rms: np.ndarray,
    frame_duration_sec: float,
    *,
    energy_quantile: float,
    threshold_scale: float,
    min_rms_threshold: float,
    min_pause_duration_sec: float,
) -> VadStats:
    """Оценить долю речи и паузы по огибающей энергии."""

    if rms.size == 0:
        return VadStats(
            speech_ratio=0.0,
            silence_ratio=1.0,
            pause_count=0,
            mean_pause_duration=0.0,
            threshold=min_rms_threshold,
        )

    base_threshold = float(np.quantile(rms, energy_quantile))
    threshold = max(base_threshold * threshold_scale, min_rms_threshold)
    active = rms >= threshold
    speech_ratio = float(np.mean(active))
    silence_ratio = 1.0 - speech_ratio

    pause_durations: list[float] = []
    pause_frames = 0
    for is_active in active:
        if is_active:
            if pause_frames > 0:
                pause_duration = pause_frames * frame_duration_sec
                if pause_duration >= min_pause_duration_sec:
                    pause_durations.append(pause_duration)
            pause_frames = 0
        else:
            pause_frames += 1

    if pause_frames > 0:
        pause_duration = pause_frames * frame_duration_sec
        if pause_duration >= min_pause_duration_sec:
            pause_durations.append(pause_duration)

    mean_pause_duration = float(np.mean(pause_durations)) if pause_durations else 0.0
    return VadStats(
        speech_ratio=round(speech_ratio, 4),
        silence_ratio=round(silence_ratio, 4),
        pause_count=len(pause_durations),
        mean_pause_duration=round(mean_pause_duration, 4),
        threshold=threshold,
    )
