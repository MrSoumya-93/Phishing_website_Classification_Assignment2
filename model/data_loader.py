"""Data loading utilities for the phishing-URL classifier."""
from pathlib import Path
from typing import Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_PATH: Path = BASE_DIR / "dataset.csv"
ID_COLUMN: str = "id"
TARGET_COLUMN: str = "Result"
PHISHING_LABEL: int = -1          # original encoding: -1 = phishing, +1 = legit


def load_and_separate_features_target(
    data_path: Path = DATA_PATH,
    id_column: str = ID_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the raw phishing-URL dataset and split into features (X) and target (y).

    Steps
    -----
    1. Read the CSV into a DataFrame.
    2. Drop the identifier column ('id') — it carries no predictive signal.
    3. Separate the target column ('Result') from the feature matrix.
    """
    raw_data = pd.read_csv(data_path).drop(columns=[id_column], errors="raise")

    if target_column not in raw_data.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    target_series = raw_data[target_column].copy()
    feature_matrix = raw_data.drop(columns=[target_column]).copy()
    return feature_matrix, target_series


def binarize_target(
    raw_target: pd.Series,
    phishing_label: int = PHISHING_LABEL,
) -> pd.Series:
    """Map {-1, +1} → {1 = phishing, 0 = legitimate} so metrics read operationally."""
    return (raw_target == phishing_label).astype(int)
