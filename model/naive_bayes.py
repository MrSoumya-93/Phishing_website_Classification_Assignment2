"""Model 4 of 5: Gaussian Naive Bayes Classifier."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline

from evaluation import evaluate_classifier, save_metrics
from splitting import MODEL_DIR, get_split

MODEL_NAME: str = "Naive Bayes"
MODEL_FILE: Path = MODEL_DIR / "naive_bayes.joblib"


def build_pipeline() -> Pipeline:
    """
    No scaler needed. GaussianNB computes the mean and variance of each
    feature per class, which is inherently scale-invariant.
    """
    return Pipeline(steps=[
        ("classifier", GaussianNB()),
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
