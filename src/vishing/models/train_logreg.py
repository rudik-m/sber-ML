"""Обучение baseline-модели LogisticRegression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from vishing.constants import CONFIGS_DIR, models_dir, reports_dir
from vishing.eval.cv import build_validation_split
from vishing.eval.metrics import compute_metrics
from vishing.eval.reports import write_training_report
from vishing.models.save_load import save_joblib, save_json, save_model_config
from vishing.models.thresholds import choose_best_threshold, extract_label_probability
from vishing.schemas import TrainingArtifacts
from vishing.text.vectorize import fit_vectorizers, get_feature_names, load_vectorizer_config, transform_texts
from vishing.utils.io import ensure_dir
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
    frame["transcript_norm"] = frame["transcript_norm"].fillna("").astype(str)
    frame = balanced_head(frame.sort_values("filepath").reset_index(drop=True), limit=limit)
    return frame


def _select_dense_columns(frame: pd.DataFrame) -> list[str]:
    numeric_columns = []
    for column in frame.columns:
        if column in IDENTIFIER_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric_columns.append(column)
    return numeric_columns


def _build_sparse_inputs(
    frame: pd.DataFrame,
    *,
    bundle,
    dense_columns: list[str],
    scaler: StandardScaler,
    fit_scaler: bool,
):
    texts = frame["transcript_norm"].fillna("").astype(str).tolist()
    text_matrix = transform_texts(bundle, texts)
    dense_values = frame[dense_columns].fillna(0.0).astype(float).to_numpy()
    scaled_dense = scaler.fit_transform(dense_values) if fit_scaler else scaler.transform(dense_values)
    dense_matrix = sparse.csr_matrix(scaled_dense)
    return sparse.hstack([text_matrix, dense_matrix], format="csr")


def train_logreg_model(features_path: Path, *, limit: int | None = None) -> TrainingArtifacts:
    """Обучить LogisticRegression, сохранить артефакты и отчёт."""

    frame = _prepare_training_frame(features_path, limit=limit)
    if len(frame) < 8:
        raise ValueError("Для обучения LogisticRegression слишком мало строк. Нужен хотя бы небольшой набор размеченных данных.")

    model_config = pd.Series(dtype=float)
    yaml_config = load_vectorizer_config(CONFIGS_DIR / "model.yaml")
    split = build_validation_split(frame, yaml_config)
    train_frame = frame.iloc[split.train_idx].reset_index(drop=True)
    val_frame = frame.iloc[split.val_idx].reset_index(drop=True)

    dense_columns = _select_dense_columns(frame)
    tfidf_config = load_vectorizer_config(CONFIGS_DIR / "text_features.yaml")
    bundle = fit_vectorizers(train_frame["transcript_norm"].tolist(), tfidf_config)
    scaler = StandardScaler()
    x_train = _build_sparse_inputs(train_frame, bundle=bundle, dense_columns=dense_columns, scaler=scaler, fit_scaler=True)
    x_val = _build_sparse_inputs(val_frame, bundle=bundle, dense_columns=dense_columns, scaler=scaler, fit_scaler=False)

    logreg_cfg = yaml_config.get("logreg", {})
    model = LogisticRegression(
        max_iter=int(logreg_cfg.get("max_iter", 4000)),
        C=float(logreg_cfg.get("C", 1.0)),
        class_weight=logreg_cfg.get("class_weight", "balanced"),
        solver=str(logreg_cfg.get("solver", "liblinear")),
    )
    y_train = train_frame["label"].to_numpy(dtype=int)
    y_val = val_frame["label"].to_numpy(dtype=int)
    model.fit(x_train, y_train)

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
    validation_predictions["model_name"] = "logreg"

    final_bundle = fit_vectorizers(frame["transcript_norm"].tolist(), tfidf_config)
    final_scaler = StandardScaler()
    x_full = _build_sparse_inputs(frame, bundle=final_bundle, dense_columns=dense_columns, scaler=final_scaler, fit_scaler=True)
    final_model = LogisticRegression(
        max_iter=int(logreg_cfg.get("max_iter", 4000)),
        C=float(logreg_cfg.get("C", 1.0)),
        class_weight=logreg_cfg.get("class_weight", "balanced"),
        solver=str(logreg_cfg.get("solver", "liblinear")),
    )
    final_model.fit(x_full, frame["label"].to_numpy(dtype=int))

    dataset_root = features_path.resolve().parents[2]
    artifact_dir = ensure_dir(models_dir(dataset_root) / "logreg")
    report_path = ensure_dir(reports_dir(dataset_root)) / "logreg_report.md"
    metrics_path = artifact_dir / "metrics.json"
    threshold_path = artifact_dir / "threshold.json"
    validation_predictions_path = artifact_dir / "validation_predictions.csv"

    save_joblib(
        {
            "model": final_model,
            "vectorizers": final_bundle,
            "scaler": final_scaler,
            "dense_columns": dense_columns,
            "classes": final_model.classes_,
        },
        artifact_dir / "model.joblib",
    )
    save_json(metrics_path, metrics)
    save_json(threshold_path, threshold_payload)
    validation_predictions.to_csv(validation_predictions_path, index=False)
    save_model_config(
        artifact_dir / "config.yaml",
        {
            "model_name": "logreg",
            "dense_columns": dense_columns,
            "random_state": split.random_state,
            "validation_rows": len(val_frame),
            "training_rows": len(train_frame),
        },
    )

    feature_names = get_feature_names(final_bundle) + [f"dense:{column}" for column in dense_columns]
    coefficients = final_model.coef_[0]
    ranked = list(zip(feature_names, coefficients))
    top_not_fraud = [item for item in ranked if item[1] > 0]
    top_not_fraud.sort(key=lambda item: item[1], reverse=True)
    top_fraud = [item for item in ranked if item[1] < 0]
    top_fraud.sort(key=lambda item: item[1])
    top_dense = [item for item in ranked if item[0].startswith("dense:")]
    top_dense.sort(key=lambda item: abs(item[1]), reverse=True)

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
        "## Топ-признаки в сторону Fraud (наиболее отрицательные коэффициенты)",
        "",
        *[f"- `{name}`: {coef:.4f}" for name, coef in top_fraud[:15]],
        "",
        "## Топ-признаки в сторону NotFraud (наиболее положительные коэффициенты)",
        "",
        *[f"- `{name}`: {coef:.4f}" for name, coef in top_not_fraud[:15]],
        "",
        "## Наиболее влиятельные dense-признаки",
        "",
        *[f"- `{name}`: {coef:.4f}" for name, coef in top_dense[:10]],
    ]
    write_training_report(report_path, "Отчёт по LogisticRegression", report_lines)

    return TrainingArtifacts(
        model_name="logreg",
        artifact_dir=artifact_dir,
        report_path=report_path,
        metrics_path=metrics_path,
        threshold_path=threshold_path,
        validation_predictions_path=validation_predictions_path,
        metrics=metrics,
        threshold=float(threshold_payload["threshold"]),
    )
