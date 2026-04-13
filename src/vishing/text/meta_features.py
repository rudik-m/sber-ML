"""Простые метапризнаки по тексту."""

from __future__ import annotations

from pathlib import Path

from vishing.text.normalize import normalize_text, tokenize_words
from vishing.utils.io import read_yaml_file
from vishing.utils.misc import safe_div


def load_text_feature_config(path: Path) -> dict:
    """Загрузить конфиг текстовых признаков."""

    return read_yaml_file(path)


def extract_text_meta_features(text: str, config: dict) -> dict[str, int | float]:
    """Извлечь базовые числовые признаки из нормализованного текста."""

    normalized_text = normalize_text(text)
    words = tokenize_words(normalized_text)
    unique_word_count = len(set(words))
    avg_word_len = safe_div(sum(len(word) for word in words), len(words))
    suspicious_words = {normalize_text(word) for word in config.get("suspicious_imperatives", [])}
    suspicious_imperative_count = sum(1 for word in words if word in suspicious_words)

    return {
        "text_char_len": len(normalized_text),
        "text_word_count": len(words),
        "unique_word_count": unique_word_count,
        "avg_word_len": round(avg_word_len, 3),
        "digit_count": sum(char.isdigit() for char in normalized_text),
        "suspicious_imperative_count": suspicious_imperative_count,
        "repetition_ratio": round(safe_div(max(len(words) - unique_word_count, 0), len(words)), 3),
        "exclamation_count": normalized_text.count("!"),
        "question_count": normalized_text.count("?"),
    }
