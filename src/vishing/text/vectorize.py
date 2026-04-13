"""TF-IDF-векторизация текстов."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from vishing.utils.io import read_yaml_file


@dataclass(slots=True)
class TfidfBundle:
    """Пара word/char TF-IDF векторизаторов."""

    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer


def load_vectorizer_config(path: Path) -> dict:
    """Загрузить конфиг TF-IDF."""

    return read_yaml_file(path)


def fit_vectorizers(texts: list[str], config: dict) -> TfidfBundle:
    """Обучить word- и char-TFIDF на тренировочном тексте."""

    word_cfg = config.get("word_tfidf", {})
    char_cfg = config.get("char_tfidf", {})

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(word_cfg.get("ngram_range", [1, 2])),
        min_df=int(word_cfg.get("min_df", 1)),
        max_features=int(word_cfg.get("max_features", 2000)),
        lowercase=False,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer=str(char_cfg.get("analyzer", "char_wb")),
        ngram_range=tuple(char_cfg.get("ngram_range", [3, 5])),
        min_df=int(char_cfg.get("min_df", 1)),
        max_features=int(char_cfg.get("max_features", 4000)),
        lowercase=False,
    )

    word_vectorizer.fit(texts)
    char_vectorizer.fit(texts)
    return TfidfBundle(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer)


def transform_texts(bundle: TfidfBundle, texts: list[str]) -> sparse.csr_matrix:
    """Преобразовать тексты в объединённую sparse-матрицу."""

    word_matrix = bundle.word_vectorizer.transform(texts)
    char_matrix = bundle.char_vectorizer.transform(texts)
    return sparse.hstack([word_matrix, char_matrix], format="csr")


def get_feature_names(bundle: TfidfBundle) -> list[str]:
    """Вернуть имена sparse-признаков в стабильном порядке."""

    word_names = [f"word:{name}" for name in bundle.word_vectorizer.get_feature_names_out()]
    char_names = [f"char:{name}" for name in bundle.char_vectorizer.get_feature_names_out()]
    return word_names + char_names


def save_vectorizers(bundle: TfidfBundle, path: Path) -> None:
    """Сохранить набор TF-IDF векторизаторов."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_vectorizers(path: Path) -> TfidfBundle:
    """Загрузить набор TF-IDF векторизаторов."""

    return joblib.load(path)
