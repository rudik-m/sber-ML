"""Простой markdown-анализ ошибок модели."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vishing.utils.io import write_text_file


def write_error_analysis(validation_frame: pd.DataFrame, path: Path, model_name: str) -> None:
    """Сохранить краткий анализ false positive / false negative."""

    if validation_frame.empty:
        write_text_file(path, "# Анализ ошибок\n\nНет validation-предсказаний для анализа.\n")
        return

    false_positive = validation_frame[
        (validation_frame["label"] == 1) & (validation_frame["predicted_label"] == 0)
    ].sort_values("fraud_score", ascending=False)
    false_negative = validation_frame[
        (validation_frame["label"] == 0) & (validation_frame["predicted_label"] == 1)
    ].sort_values("fraud_score", ascending=True)

    lines = [
        "# Анализ ошибок",
        "",
        f"- Модель: `{model_name}`",
        f"- Всего validation-строк: {len(validation_frame)}",
        f"- False Positive (NotFraud -> Fraud): {len(false_positive)}",
        f"- False Negative (Fraud -> NotFraud): {len(false_negative)}",
        "",
        "## Наиболее уверенные false positive",
        "",
    ]

    for _, row in false_positive.head(5).iterrows():
        lines.extend(
            [
                f"- `{row['filepath']}` | score={row['fraud_score']:.3f} | триггеры={row.get('top_trigger_categories', '')}",
                f"  Текст: {str(row.get('transcript_norm', ''))[:240]}",
            ]
        )

    lines.extend(["", "## Наиболее уверенные false negative", ""])
    for _, row in false_negative.head(5).iterrows():
        lines.extend(
            [
                f"- `{row['filepath']}` | score={row['fraud_score']:.3f} | триггеры={row.get('top_trigger_categories', '')}",
                f"  Текст: {str(row.get('transcript_norm', ''))[:240]}",
            ]
        )

    write_text_file(path, "\n".join(lines) + "\n")
