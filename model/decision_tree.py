"""Model 2 of 5: Decision Tree Classifier."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from evaluation import evaluate_classifier
from splitting import MODEL_DIR, get_split

RANDOM_STATE: int = 42
MODEL_NAME: str = "Decision Tree"
MODEL_FILE: Path = MODEL_DIR / "decision_tree.joblib"


def build_pipeline() -> Pipeline:
    """Single-step pipeline: trees are scale-invariant, no scaler needed."""
    return Pipeline(steps=[
        ("classifier", DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=5,
            min_samples_split=5,
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

    # --- Top-10 feature importances (useful for the README observation) ---
    importances = (
        pd.Series(
            pipeline.named_steps["classifier"].feature_importances_,
            index=X_train.columns,
        )
        .sort_values(ascending=False)
        .head(10)
        .round(4)
    )
    print("\n=== Top-10 Feature Importances ===")
    print(importances.to_string())

    # --- Persist ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE, compress=3)
    print(f"\nModel saved -> {MODEL_FILE}")
