"""Простой ансамбль вероятностей двух моделей."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vishing.constants import CONFIGS_DIR, models_dir, reports_dir
from vishing.eval.metrics import compute_metrics
from vishing.eval.reports import write_training_report
from vishing.models.save_load import load_json, save_json, save_model_config
from vishing.models.thresholds import choose_best_threshold
from vishing.schemas import TrainingArtifacts
from vishing.utils.io import ensure_dir, read_yaml_file


def combine_probabilities(prob_logreg: pd.Series, prob_catboost: pd.Series, *, weight_logreg: float, weight_catboost: float) -> pd.Series:
    """Скомбинировать fraud score двух моделей."""

    total = weight_logreg + weight_catboost
    return (prob_logreg * weight_logreg + prob_catboost * weight_catboost) / total


def build_ensemble(dataset_root: Path) -> TrainingArtifacts:
    """Построить и сохранить ансамбль поверх validation-предсказаний."""

    yaml_config = read_yaml_file(CONFIGS_DIR / "model.yaml")
    ensemble_cfg = yaml_config.get("ensemble", {})
    weight_logreg = float(ensemble_cfg.get("weight_logreg", 0.7))
    weight_catboost = float(ensemble_cfg.get("weight_catboost", 0.3))

    models_root = models_dir(dataset_root)
    logreg_val = pd.read_csv(models_root / "logreg" / "validation_predictions.csv")
    catboost_val = pd.read_csv(models_root / "catboost" / "validation_predictions.csv")

    merged = logreg_val.rename(columns={"fraud_score": "fraud_score_logreg"}).merge(
        catboost_val[["filepath", "fraud_score"]].rename(columns={"fraud_score": "fraud_score_catboost"}),
        on="filepath",
        how="inner",
    )
    if merged.empty:
        raise ValueError("Не удалось собрать ансамбль: validation-предсказания моделей не пересекаются.")

    merged["fraud_score"] = combine_probabilities(
        merged["fraud_score_logreg"],
        merged["fraud_score_catboost"],
        weight_logreg=weight_logreg,
        weight_catboost=weight_catboost,
    )

    y_true = merged["label"].astype(int).to_numpy()
    threshold_payload = choose_best_threshold(y_true, merged["fraud_score"].to_numpy())
    metrics = compute_metrics(y_true, merged["fraud_score"].to_numpy(), threshold=float(threshold_payload["threshold"]))
    merged["predicted_label"] = (merged["fraud_score"] < float(threshold_payload["threshold"])).astype(int)
    merged["threshold"] = float(threshold_payload["threshold"])
    merged["model_name"] = "ensemble"

    artifact_dir = ensure_dir(models_root / "ensemble")
    report_path = ensure_dir(reports_dir(dataset_root)) / "ensemble_report.md"
    metrics_path = artifact_dir / "metrics.json"
    threshold_path = artifact_dir / "threshold.json"
    validation_predictions_path = artifact_dir / "validation_predictions.csv"

    save_json(metrics_path, metrics)
    save_json(threshold_path, threshold_payload)
    merged.to_csv(validation_predictions_path, index=False)
    save_model_config(
        artifact_dir / "config.yaml",
        {
            "model_name": "ensemble",
            "weight_logreg": weight_logreg,
            "weight_catboost": weight_catboost,
            "logreg_threshold": load_json(models_root / "logreg" / "threshold.json").get("threshold"),
            "catboost_threshold": load_json(models_root / "catboost" / "threshold.json").get("threshold"),
        },
    )

    report_lines = [
        f"- Вес LogisticRegression: {weight_logreg}",
        f"- Вес CatBoost: {weight_catboost}",
        f"- Accuracy: {metrics['accuracy']}",
        f"- Precision (Fraud): {metrics['precision']}",
        f"- Recall (Fraud): {metrics['recall']}",
        f"- F1 (Fraud): {metrics['f1']}",
        f"- ROC-AUC: {metrics['roc_auc']}",
        f"- Итоговый порог Fraud: {threshold_payload['threshold']}",
    ]
    write_training_report(report_path, "Отчёт по ансамблю", report_lines)

    return TrainingArtifacts(
        model_name="ensemble",
        artifact_dir=artifact_dir,
        report_path=report_path,
        metrics_path=metrics_path,
        threshold_path=threshold_path,
        validation_predictions_path=validation_predictions_path,
        metrics=metrics,
        threshold=float(threshold_payload["threshold"]),
    )
