"""Assignment 2 | Model 5 of 5: Support Vector Machine."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluation import evaluate_classifier, save_metrics
from splitting import MODEL_DIR, get_split

RANDOM_STATE: int = 42
MODEL_NAME: str = "Support Vector Machine"
MODEL_FILE: Path = MODEL_DIR / "svm.joblib"


def build_pipeline() -> Pipeline:
    """
    Scaler is mandatory for SVM (especially with RBF kernel).
    probability=False (default) is faster; we use decision_function for AUC.
    """
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=20,
            gamma="scale",
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

    # --- Labelled confusion matrix ---
    cm = confusion_matrix(y_test, pipeline.predict(X_test), labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Legitimate (0)", "Actual: Phishing (1)"],
        columns=["Predicted: Legitimate (0)", "Predicted: Phishing (1)"],
    )
    print("\n=== Confusion Matrix ===")
    print(cm_df.to_string())

    # --- Persist ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE, compress=3)
    save_metrics(MODEL_NAME, result)
    print(f"\nModel saved -> {MODEL_FILE}")
