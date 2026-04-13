"""Инференс для logreg, catboost и ансамбля."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy import sparse

from vishing.constants import PREDICTIONS_CSV_NAME, PREDICTIONS_DEBUG_CSV_NAME, models_dir, predictions_dir
from vishing.models.ensemble import combine_probabilities
from vishing.models.save_load import load_joblib, load_json, load_model_config
from vishing.models.thresholds import extract_label_probability, score_to_label
from vishing.schemas import PredictionArtifacts
from vishing.text.vectorize import transform_texts
from vishing.utils.io import ensure_dir, save_dataframe_csv


def _prepare_dense_frame(frame: pd.DataFrame, dense_columns: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for column in dense_columns:
        if column not in prepared:
            prepared[column] = 0.0
    return prepared[dense_columns].fillna(0.0).astype(float)


def predict_logreg(feature_frame: pd.DataFrame, dataset_root: Path) -> tuple[np.ndarray, float]:
    """Получить fraud score от LogisticRegression."""

    artifact_dir = models_dir(dataset_root) / "logreg"
    bundle = load_joblib(artifact_dir / "model.joblib")
    threshold = float(load_json(artifact_dir / "threshold.json")["threshold"])
    texts = feature_frame["transcript_norm"].fillna("").astype(str).tolist()
    text_matrix = transform_texts(bundle["vectorizers"], texts)
    dense_frame = _prepare_dense_frame(feature_frame, bundle["dense_columns"])
    dense_values = bundle["scaler"].transform(dense_frame.to_numpy())
    matrix = sparse.hstack([text_matrix, sparse.csr_matrix(dense_values)], format="csr")
    probabilities = bundle["model"].predict_proba(matrix)
    fraud_scores = extract_label_probability(probabilities, bundle["classes"], label=0)
    return fraud_scores, threshold


def predict_catboost(feature_frame: pd.DataFrame, dataset_root: Path) -> tuple[np.ndarray, float]:
    """Получить fraud score от CatBoost."""

    artifact_dir = models_dir(dataset_root) / "catboost"
    config = load_model_config(artifact_dir / "config.yaml")
    threshold = float(load_json(artifact_dir / "threshold.json")["threshold"])
    dense_columns = list(config.get("dense_columns", []))
    dense_frame = _prepare_dense_frame(feature_frame, dense_columns)
    model = CatBoostClassifier()
    model.load_model(str(artifact_dir / "model.cbm"))
    probabilities = model.predict_proba(dense_frame)
    fraud_scores = extract_label_probability(probabilities, model.classes_, label=0)
    return fraud_scores, threshold


def predict_model(feature_frame: pd.DataFrame, dataset_root: Path, model_name: str) -> tuple[np.ndarray, float]:
    """Унифицированный интерфейс предсказания fraud score."""

    normalized_name = model_name.lower()
    if normalized_name == "logreg":
        return predict_logreg(feature_frame, dataset_root)
    if normalized_name == "catboost":
        return predict_catboost(feature_frame, dataset_root)
    if normalized_name == "ensemble":
        artifact_dir = models_dir(dataset_root) / "ensemble"
        config = load_model_config(artifact_dir / "config.yaml")
        threshold = float(load_json(artifact_dir / "threshold.json")["threshold"])
        logreg_scores, _ = predict_logreg(feature_frame, dataset_root)
        catboost_scores, _ = predict_catboost(feature_frame, dataset_root)
        ensemble_scores = combine_probabilities(
            pd.Series(logreg_scores),
            pd.Series(catboost_scores),
            weight_logreg=float(config.get("weight_logreg", 0.7)),
            weight_catboost=float(config.get("weight_catboost", 0.3)),
        ).to_numpy()
        return ensemble_scores, threshold
    raise ValueError(f"Неизвестная модель: {model_name}")


def save_predictions(feature_frame: pd.DataFrame, fraud_scores: np.ndarray, threshold: float, dataset_root: Path, model_name: str) -> PredictionArtifacts:
    """Сохранить основной и расширенный CSV с предсказаниями."""

    predictions_root = ensure_dir(predictions_dir(dataset_root))
    predicted_labels = np.array([score_to_label(float(score), threshold) for score in fraud_scores], dtype=int)

    main_frame = feature_frame[["filename"]].copy()
    main_frame["label"] = predicted_labels
    debug_frame = feature_frame[["filepath", "filename", "transcript_norm", "top_trigger_categories", "matched_trigger_phrases"]].copy()
    debug_frame["fraud_score"] = fraud_scores
    debug_frame["predicted_label"] = predicted_labels
    debug_frame["matched_phrases"] = debug_frame.pop("matched_trigger_phrases")
    debug_frame["threshold"] = threshold
    debug_frame["model_name"] = model_name

    predictions_path = predictions_root / PREDICTIONS_CSV_NAME
    debug_path = predictions_root / PREDICTIONS_DEBUG_CSV_NAME
    save_dataframe_csv(main_frame, predictions_path)
    save_dataframe_csv(debug_frame, debug_path)
    return PredictionArtifacts(predictions_path=predictions_path, debug_predictions_path=debug_path, row_count=len(main_frame))
