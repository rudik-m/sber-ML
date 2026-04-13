"""Regex-паттерны для шумного ASR-текста."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from vishing.text.normalize import normalize_text
from vishing.utils.io import read_yaml_file
from vishing.utils.misc import semicolon_join, top_counts_to_text


@dataclass(frozen=True, slots=True)
class PatternRule:
    """Одна regex-правило для обнаружения паттерна."""

    name: str
    category: str
    description: str
    regex: str
    compiled: re.Pattern[str]


@dataclass(slots=True)
class PatternMatchSummary:
    """Результат работы набора regex-паттернов."""

    features: dict[str, int | float | str]
    matched_pattern_names: list[str]
    category_counts: dict[str, int]


def load_pattern_catalog(path: Path) -> dict[str, PatternRule]:
    """Загрузить и скомпилировать паттерны из YAML."""

    payload = read_yaml_file(path)
    patterns = payload.get("patterns", {})
    catalog: dict[str, PatternRule] = {}
    for name, raw_pattern in patterns.items():
        category = str(raw_pattern.get("category", "other"))
        description = str(raw_pattern.get("description", ""))
        regex = str(raw_pattern.get("regex", "")).strip()
        if not regex:
            continue
        catalog[name] = PatternRule(
            name=name,
            category=category,
            description=description,
            regex=regex,
            compiled=re.compile(regex, flags=re.IGNORECASE | re.UNICODE),
        )
    return catalog


def extract_pattern_features(text: str, catalog: dict[str, PatternRule]) -> PatternMatchSummary:
    """Извлечь regex-признаки из текста."""

    normalized_text = normalize_text(text)
    features: dict[str, int | float | str] = {}
    matched_pattern_names: list[str] = []
    category_counts: dict[str, int] = {}
    total_count = 0
    for name, rule in catalog.items():
        count = len(rule.compiled.findall(normalized_text))
        features[f"pattern_{name}_count"] = count
        features[f"has_pattern_{name}"] = int(count > 0)
        category_counts[rule.category] = category_counts.get(rule.category, 0) + count
        total_count += count
        if count > 0:
            matched_pattern_names.append(name)

    for category, count in category_counts.items():
        features[f"pattern_category_{category}_count"] = count

    features["pattern_total_count"] = total_count
    features["pattern_unique_count"] = len(set(matched_pattern_names))
    features["matched_pattern_names"] = semicolon_join(sorted(set(matched_pattern_names)))
    features["top_pattern_categories"] = top_counts_to_text(category_counts)

    return PatternMatchSummary(
        features=features,
        matched_pattern_names=sorted(set(matched_pattern_names)),
        category_counts=category_counts,
    )
