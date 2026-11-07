"""Model 1 of 5: Logistic Regression."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import evaluate_classifier
from splitting import MODEL_DIR, get_split

RANDOM_STATE: int = 42
MODEL_NAME: str = "Logistic Regression"
MODEL_FILE: Path = MODEL_DIR / "logistic_regression.joblib"


def build_pipeline() -> Pipeline:
    """Scaler + classifier bundled so the Streamlit app loads a single artifact."""
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_split()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # --- Metrics ---
    result = evaluate_classifier(MODEL_NAME, pipeline, X_test, y_test)
    print("\n=== Evaluation Metrics ===")
    print(pd.DataFrame([result]).set_index("ML Model Name").T.to_string(header=False))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, pipeline.predict(X_test))
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    # --- Persist ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE, compress=3)
    print(f"\nModel saved -> {MODEL_FILE}")
