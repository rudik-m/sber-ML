"""Метрики качества для бинарной задачи Fraud/NotFraud."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from vishing.models.thresholds import apply_threshold


def compute_metrics(y_true: np.ndarray, fraud_scores: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    """Посчитать набор основных метрик на validation."""

    predicted_labels = apply_threshold(fraud_scores, threshold)
    fraud_true = (y_true == 0).astype(int)
    fraud_pred = (predicted_labels == 0).astype(int)

    roc_auc: float | None
    try:
        roc_auc = round(float(roc_auc_score(fraud_true, fraud_scores)), 6)
    except Exception:
        roc_auc = None

    return {
        "accuracy": round(float(accuracy_score(y_true, predicted_labels)), 6),
        "precision": round(float(precision_score(fraud_true, fraud_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(fraud_true, fraud_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(fraud_true, fraud_pred, zero_division=0)), 6),
        "roc_auc": roc_auc,
        "fraud_count": int(np.sum(y_true == 0)),
        "not_fraud_count": int(np.sum(y_true == 1)),
    }
