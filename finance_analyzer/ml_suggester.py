from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _build_text(df: pd.DataFrame) -> pd.Series:
    desc = df["description"].fillna("")
    if "merchant" in df.columns:
        merch = df["merchant"].fillna("")
        text = merch + " | " + desc
    else:
        text = desc
    return text.str.lower()


def train_category_suggester(df: pd.DataFrame) -> Optional[Tuple[TfidfVectorizer, LogisticRegression]]:
    """
    Train a simple text → category model using existing labeled data.
    Returns (vectorizer, model) or None if not enough data.
    """
    df_train = df.copy()
    df_train = df_train[df_train["category"].notna()]
    df_train = df_train[df_train["category"] != "Uncategorized"]

    if df_train.empty or df_train["category"].nunique() < 2:
        return None

    X_text = _build_text(df_train)
    y = df_train["category"].values

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vectorizer.fit_transform(X_text)

    model = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
    )
    model.fit(X, y)

    return vectorizer, model


def suggest_category_for_single(
    description: str,
    merchant: str | None,
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
) -> Tuple[str, float]:
    """
    Return (predicted_category, probability) for a single transaction.
    """
    if merchant is None:
        merchant = ""

    text = (merchant + " | " + description).lower()
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    idx = np.argmax(proba)
    pred = model.classes_[idx]
    confidence = float(proba[idx])
    return pred, confidence
