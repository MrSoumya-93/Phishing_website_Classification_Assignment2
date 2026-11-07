"""Train/test split + test-data export — shared by all five models."""
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import (
    PHISHING_LABEL,
    TARGET_COLUMN,
    binarize_target,
    load_and_separate_features_target,
)

MODEL_DIR: Path = Path("model")
TEST_DATA_PATH: Path = Path("test_data.csv")
SCHEMA_PATH: Path = MODEL_DIR / "feature_schema.json"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20


def _export_test_artifacts(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: List[str],
) -> None:
    """Write test_data.csv (features + target in ORIGINAL -1/+1 encoding) + schema."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    test_export = X_test.copy()
    test_export[TARGET_COLUMN] = y_test.map({1: PHISHING_LABEL, 0: 1}).values
    test_export.to_csv(TEST_DATA_PATH, index=False)

    SCHEMA_PATH.write_text(json.dumps({
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "phishing_label": PHISHING_LABEL,
    }, indent=2))


def get_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load, binarize and split with a fixed seed. Also exports test_data.csv and
    feature_schema.json for the Streamlit app.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X, raw_y = load_and_separate_features_target()
    y = binarize_target(raw_y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    _export_test_artifacts(X_test, y_test, X.columns.tolist())
    return X_train, X_test, y_train, y_test
