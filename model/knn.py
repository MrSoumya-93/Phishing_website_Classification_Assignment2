"""Model 3 of 5: K-Nearest Neighbor Classifier."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import evaluate_classifier, save_metrics
from splitting import MODEL_DIR, get_split

MODEL_NAME: str = "K-Nearest Neighbors"
MODEL_FILE: Path = MODEL_DIR / "knn.joblib"


def build_pipeline() -> Pipeline:
    """
    Scaler is mandatory for KNN: distances are computed in feature space,
    so an unscaled feature would dominate the neighbourhood search.
    """
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=10,
            weights="distance",   # closer neighbours get more vote weight
            n_jobs=-1,
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
