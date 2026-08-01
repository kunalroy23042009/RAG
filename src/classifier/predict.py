"""
Module 1, step 3 — predict a category for a new notice.

This is what the main pipeline calls for every newly extracted notice.
The trained model is loaded once and cached in memory.

Usage (from other code):
    from src.classifier.predict import predict_category
    category = predict_category(subject_line, main_body_content)
"""
import os
from typing import Optional

import joblib

from config import CLASSIFIER_MODEL_PATH

_model = None
_model_loaded = False


def _get_model():
    global _model, _model_loaded
    if not _model_loaded:
        if os.path.exists(CLASSIFIER_MODEL_PATH):
            _model = joblib.load(CLASSIFIER_MODEL_PATH)
        _model_loaded = True
    return _model


def predict_category(subject_line: Optional[str], main_body_content: Optional[str]) -> str:
    """
    Returns a predicted category string, or "Uncategorized" if no model has
    been trained yet — this lets the main pipeline run before Module 1 is
    fully set up without crashing.
    """
    model = _get_model()
    if model is None:
        return "Uncategorized"

    text = f"{subject_line or ''} {main_body_content or ''}"
    return model.predict([text])[0]
