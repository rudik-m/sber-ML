"""Обучение основной модели CatBoostClassifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from vishing.constants import CONFIGS_DIR, models_dir, reports_dir
from vishing.eval.cv import build_validation_split
from vishing.eval.metrics import compute_metrics
from vishing.eval.reports import write_training_report
from vishing.models.save_load import save_json, save_model_config
from vishing.models.thresholds import choose_best_threshold, extract_label_probability
from vishing.schemas import TrainingArtifacts
from vishing.utils.io import ensure_dir, read_yaml_file
from vishing.utils.misc import balanced_head

IDENTIFIER_COLUMNS = {
    "filepath",
    "filename",
    "split",
    "label_dir",
    "label",
    "group_id",
    "transcript_raw",
    "transcript_norm",
    "language",
    "transcript_error",
    "matched_trigger_phrases",
    "matched_pattern_names",
    "top_trigger_categories",
    "top_pattern_categories",
    "audio_feature_error",
}


def _prepare_training_frame(features_path: Path, limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(features_path)
    frame = frame[frame["label"].notna()].copy()
    frame["label"] = frame["label"].astype(int)
    frame = balanced_head(frame.sort_values("filepath").reset_index(drop=True), limit=limit)
    return frame


def _select_dense_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in IDENTIFIER_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def train_catboost_model(features_path: Path, *, limit: int | None = None) -> TrainingArtifacts:
    """Обучить CatBoost на dense handcrafted-признаках."""

    frame = _prepare_training_frame(features_path, limit=limit)
    if len(frame) < 8:
        raise ValueError("Для обучения CatBoost слишком мало строк. Нужен хотя бы небольшой набор размеченных данных.")

    yaml_config = read_yaml_file(CONFIGS_DIR / "model.yaml")
    split = build_validation_split(frame, yaml_config)
    train_frame = frame.iloc[split.train_idx].reset_index(drop=True)
    val_frame = frame.iloc[split.val_idx].reset_index(drop=True)
    dense_columns = _select_dense_columns(frame)

    x_train = train_frame[dense_columns]
    x_val = val_frame[dense_columns]
    y_train = train_frame["label"].to_numpy(dtype=int)
    y_val = val_frame["label"].to_numpy(dtype=int)

    cat_cfg = yaml_config.get("catboost", {})
    model = CatBoostClassifier(
        iterations=int(cat_cfg.get("iterations", 300)),
        depth=int(cat_cfg.get("depth", 4)),
        learning_rate=float(cat_cfg.get("learning_rate", 0.05)),
        l2_leaf_reg=float(cat_cfg.get("l2_leaf_reg", 3.0)),
        loss_function=str(cat_cfg.get("loss_function", "Logloss")),
        eval_metric=str(cat_cfg.get("eval_metric", "AUC")),
        verbose=bool(cat_cfg.get("verbose", False)),
        random_seed=int(yaml_config.get("training", {}).get("random_state", 42)),
        allow_writing_files=False,
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=False)
    val_probabilities = model.predict_proba(x_val)
    fraud_scores_val = extract_label_probability(val_probabilities, model.classes_, label=0)
    threshold_payload = choose_best_threshold(y_val, fraud_scores_val)
    metrics = compute_metrics(y_val, fraud_scores_val, threshold=float(threshold_payload["threshold"]))

    validation_predictions = val_frame[
        ["filepath", "filename", "label", "transcript_norm", "top_trigger_categories", "matched_trigger_phrases"]
    ].copy()
    validation_predictions["fraud_score"] = fraud_scores_val
    validation_predictions["predicted_label"] = (fraud_scores_val < float(threshold_payload["threshold"])).astype(int)
    validation_predictions["threshold"] = float(threshold_payload["threshold"])
    validation_predictions["model_name"] = "catboost"

    final_model = CatBoostClassifier(
        iterations=int(cat_cfg.get("iterations", 300)),
        depth=int(cat_cfg.get("depth", 4)),
        learning_rate=float(cat_cfg.get("learning_rate", 0.05)),
        l2_leaf_reg=float(cat_cfg.get("l2_leaf_reg", 3.0)),
        loss_function=str(cat_cfg.get("loss_function", "Logloss")),
        eval_metric=str(cat_cfg.get("eval_metric", "AUC")),
        verbose=bool(cat_cfg.get("verbose", False)),
        random_seed=int(yaml_config.get("training", {}).get("random_state", 42)),
        allow_writing_files=False,
    )
    final_model.fit(frame[dense_columns], frame["label"].to_numpy(dtype=int), verbose=False)

    dataset_root = features_path.resolve().parents[2]
    artifact_dir = ensure_dir(models_dir(dataset_root) / "catboost")
    report_path = ensure_dir(reports_dir(dataset_root)) / "catboost_report.md"
    metrics_path = artifact_dir / "metrics.json"
    threshold_path = artifact_dir / "threshold.json"
    validation_predictions_path = artifact_dir / "validation_predictions.csv"

    final_model.save_model(str(artifact_dir / "model.cbm"))
    save_json(metrics_path, metrics)
    save_json(threshold_path, threshold_payload)
    validation_predictions.to_csv(validation_predictions_path, index=False)
    save_model_config(
        artifact_dir / "config.yaml",
        {
            "model_name": "catboost",
            "dense_columns": dense_columns,
            "random_state": split.random_state,
            "validation_rows": len(val_frame),
            "training_rows": len(train_frame),
        },
    )

    importances = final_model.get_feature_importance(prettified=True)
    importances.to_csv(artifact_dir / "feature_importance.csv", index=False)
    top_features = importances.head(15).to_dict(orient="records")
    report_lines = [
        f"- Размер train: {len(train_frame)}",
        f"- Размер validation: {len(val_frame)}",
        f"- Accuracy: {metrics['accuracy']}",
        f"- Precision (Fraud): {metrics['precision']}",
        f"- Recall (Fraud): {metrics['recall']}",
        f"- F1 (Fraud): {metrics['f1']}",
        f"- ROC-AUC: {metrics['roc_auc']}",
        f"- Порог Fraud: {threshold_payload['threshold']}",
        "",
        "## Топ feature importance",
        "",
        *[f"- `{row['Feature Id']}`: {row['Importances']:.4f}" for row in top_features],
    ]
    write_training_report(report_path, "Отчёт по CatBoost", report_lines)

    return TrainingArtifacts(
        model_name="catboost",
        artifact_dir=artifact_dir,
        report_path=report_path,
        metrics_path=metrics_path,
        threshold_path=threshold_path,
        validation_predictions_path=validation_predictions_path,
        metrics=metrics,
        threshold=float(threshold_payload["threshold"]),
    )
