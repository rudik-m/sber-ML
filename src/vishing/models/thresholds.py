"""Работа с порогом и переводом вероятности в метку."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def score_to_label(fraud_score: float, threshold: float) -> int:
    """Перевести fraud score в итоговую метку задачи."""

    return 0 if fraud_score >= threshold else 1


def apply_threshold(fraud_scores: np.ndarray, threshold: float) -> np.ndarray:
    """Применить порог ко всему массиву вероятностей."""

    return np.array([score_to_label(float(score), threshold) for score in fraud_scores], dtype=int)


def extract_label_probability(probabilities: np.ndarray, classes: np.ndarray, label: int = 0) -> np.ndarray:
    """Достать вероятность нужного класса из `predict_proba`."""

    class_to_index = {int(class_id): idx for idx, class_id in enumerate(classes.tolist())}
    if label not in class_to_index:
        raise KeyError(f"В `predict_proba` нет класса {label}. Доступны: {sorted(class_to_index)}")
    return probabilities[:, class_to_index[label]]


def choose_best_threshold(y_true: np.ndarray, fraud_scores: np.ndarray) -> dict[str, float]:
    """Подобрать порог по F1 для класса Fraud."""

    fraud_true = (y_true == 0).astype(int)
    candidate_thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidate_thresholds:
        pred_fraud = (fraud_scores >= threshold).astype(int)
        score = f1_score(fraud_true, pred_fraud, zero_division=0)
        if score > best_score or (abs(score - best_score) < 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_threshold = float(threshold)
            best_score = float(score)

    return {"threshold": round(best_threshold, 4), "fraud_f1": round(best_score, 6)}
