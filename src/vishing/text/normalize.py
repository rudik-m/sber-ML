"""Нормализация текста после ASR."""

from __future__ import annotations

import re

_SPACE_RE = re.compile(r"\s+")
_TECH_RE = re.compile(r"[\[\]{}<>|*_#^~]+")
_NOISE_RE = re.compile(r"[“”\"`]+")
_DASH_RE = re.compile(r"[–—]+")
_SMS_RE = re.compile(r"\bс\s*м\s*с\b", flags=re.IGNORECASE)
_PUNCT_RE = re.compile(r"([!?.,;:]){2,}")
_NON_TEXT_RE = re.compile(r"[^\w\s!?.,:%;-]+", flags=re.UNICODE)
_WORD_RE = re.compile(r"[0-9a-zA-Zа-яА-Я]+", flags=re.UNICODE)


def maybe_fix_common_asr_artifacts(text: str) -> str:
    """Исправить несколько типовых артефактов ASR."""

    fixed = text.replace("ё", "е").replace("Ё", "Е")
    fixed = _SMS_RE.sub("смс", fixed)
    fixed = _NOISE_RE.sub(" ", fixed)
    fixed = _DASH_RE.sub("-", fixed)
    fixed = fixed.replace("_", " ")
    fixed = _PUNCT_RE.sub(r"\1", fixed)
    return fixed


def normalize_text(text: str) -> str:
    """Привести текст к компактному нормализованному виду."""

    normalized = maybe_fix_common_asr_artifacts(text or "")
    normalized = normalized.lower().strip()
    normalized = _TECH_RE.sub(" ", normalized)
    normalized = _NON_TEXT_RE.sub(" ", normalized)
    normalized = _SPACE_RE.sub(" ", normalized)
    return normalized.strip()


def tokenize_words(text: str) -> list[str]:
    """Разбить текст на слова после нормализации."""

    return _WORD_RE.findall(normalize_text(text))
