"""Словарь триггеров и извлечение соответствующих признаков."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from vishing.text.normalize import normalize_text, tokenize_words
from vishing.utils.io import read_yaml_file
from vishing.utils.misc import safe_div, semicolon_join, top_counts_to_text


@dataclass(frozen=True, slots=True)
class TriggerCategory:
    """Одна категория словаря триггеров."""

    name: str
    description: str
    weight: float
    phrases: list[str]
    patterns: list[re.Pattern[str]]


@dataclass(slots=True)
class TriggerMatchSummary:
    """Результат поиска триггеров в одном тексте."""

    features: dict[str, int | float | str]
    matched_phrases: list[str]
    category_counts: dict[str, int]


def _compile_phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.UNICODE)


def load_trigger_catalog(path: Path) -> dict[str, TriggerCategory]:
    """Загрузить YAML-конфиг триггеров."""

    payload = read_yaml_file(path)
    categories = payload.get("categories", {})
    catalog: dict[str, TriggerCategory] = {}
    for name, raw_category in categories.items():
        description = str(raw_category.get("description", "")).strip()
        weight = float(raw_category.get("weight", 1.0))
        raw_phrases = raw_category.get("phrases", []) + raw_category.get("aliases", [])
        phrases = []
        for phrase in raw_phrases:
            normalized = normalize_text(str(phrase))
            if normalized and normalized not in phrases:
                phrases.append(normalized)

        catalog[name] = TriggerCategory(
            name=name,
            description=description,
            weight=weight,
            phrases=phrases,
            patterns=[_compile_phrase_pattern(phrase) for phrase in phrases],
        )

    return catalog


def extract_trigger_features(text: str, catalog: dict[str, TriggerCategory]) -> TriggerMatchSummary:
    """Построить признаки по словарю триггеров."""

    normalized_text = normalize_text(text)
    words = tokenize_words(normalized_text)
    word_count = len(words)

    matched_phrases: list[str] = []
    category_counts: dict[str, int] = {}
    weighted_score = 0.0
    phrase_count_map: dict[str, int] = {}
    for category_name, category in catalog.items():
        category_total = 0
        for phrase, pattern in zip(category.phrases, category.patterns):
            count = len(pattern.findall(normalized_text))
            if count > 0:
                phrase_count_map[phrase] = phrase_count_map.get(phrase, 0) + count
                matched_phrases.append(phrase)
                category_total += count
        category_counts[category_name] = category_total
        weighted_score += category_total * category.weight

    unique_phrases = sorted(set(matched_phrases))
    total_count = sum(category_counts.values())

    has_safe_account = any(phrase in {"безопасный счет", "резервный счет", "защитный счет"} for phrase in unique_phrases)
    has_sms_code = any(phrase in {"код из смс", "смс код", "код подтверждения", "назовите код"} for phrase in unique_phrases)
    has_bank_security_phrase = bool(category_counts.get("authority", 0))
    has_dont_hang_up = "не кладите трубку" in unique_phrases or "оставайтесь на линии" in unique_phrases
    has_credit_phrase = bool(category_counts.get("credit", 0))

    features: dict[str, int | float | str] = {
        "trigger_total_count": total_count,
        "trigger_unique_count": len(unique_phrases),
        "trigger_weighted_score": round(weighted_score, 3),
        "trigger_density_per_100_words": round(safe_div(total_count * 100.0, word_count), 3),
        "authority_count": category_counts.get("authority", 0),
        "money_count": category_counts.get("money", 0),
        "verification_count": category_counts.get("verification", 0),
        "urgency_count": category_counts.get("urgency", 0),
        "credit_count": category_counts.get("credit", 0),
        "sim_count": category_counts.get("sim", 0),
        "relative_emergency_count": category_counts.get("relative_emergency", 0),
        "has_safe_account": int(has_safe_account),
        "has_sms_code": int(has_sms_code),
        "has_bank_security_phrase": int(has_bank_security_phrase),
        "has_dont_hang_up": int(has_dont_hang_up),
        "has_credit_phrase": int(has_credit_phrase),
        "money_and_urgency": int(category_counts.get("money", 0) > 0 and category_counts.get("urgency", 0) > 0),
        "authority_and_verification": int(
            category_counts.get("authority", 0) > 0 and category_counts.get("verification", 0) > 0
        ),
        "credit_and_sms": int(category_counts.get("credit", 0) > 0 and has_sms_code),
        "safe_account_and_transfer": int(has_safe_account and category_counts.get("money", 0) > 0),
        "matched_trigger_phrases": semicolon_join(unique_phrases),
        "top_trigger_categories": top_counts_to_text(category_counts),
    }
    return TriggerMatchSummary(features=features, matched_phrases=unique_phrases, category_counts=category_counts)
