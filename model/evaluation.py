"""Shared evaluation metrics — used by all five assignment models."""
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

def evaluate_classifier(
    model_name: str,
    fitted_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Compute Accuracy, AUC, Precision, Recall, F1 and MCC for a fitted model."""
    y_pred = fitted_model.predict(X_test)
    try:
        y_score = fitted_model.predict_proba(X_test)[:, 1]
    except (AttributeError, NotImplementedError):
        y_score = fitted_model.decision_function(X_test)

    return {
        "ML Model Name": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

def save_metrics(
    model_name: str,
    metrics: Dict[str, Any],
    out_dir: Path = Path("model"),
) -> Path:
    """Persist a model's metrics dict as model/<slug>_metrics.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model_name.lower().replace(" (", "_").replace(")", "").replace(" ", "_")
    path = out_dir / f"{slug}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path
