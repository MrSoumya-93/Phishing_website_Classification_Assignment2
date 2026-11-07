import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay
)

# ---------------------------------------------------------------------------
# Configuration & State
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Phishing URL Classifier", layout="wide", page_icon="🛡️")

MODEL_DIR = Path("model")
PHISHING_LABEL = -1

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

@st.cache_resource
def load_model(model_name: str):
    model_path = MODEL_DIR / MODEL_FILES[model_name]
    return joblib.load(model_path)

@st.cache_data
def load_schema():
    with open(MODEL_DIR / "feature_schema.json", "r") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
st.title("🛡️ Phishing Website Classification")
st.markdown("""
This Streamlit app evaluates pre-trained models on the Phishing Website dataset, including Logistic Regression, Decision Tree, KNN, Naive Bayes, and SVM. It reports comprehensive metrics (Accuracy, AUC, Precision, Recall, F1 Score,
 MCC) and visualizes confusion matrices to highlight strengths and weaknesses in phishing detection.


**Instructions:** Upload the `test_data.csv` file provided in the GitHub repository.
""")

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Choose a classifier:", list(MODEL_FILES.keys()))

uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    schema = load_schema()
    feature_cols = schema["feature_columns"]
    target_col = schema["target_column"]
    
    # 1. Validate Target Column
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in the uploaded file.")
        
    # 2. Validate Feature Columns
    elif not all(col in df.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in df.columns]
        st.error(f"Missing feature columns! Please ensure you uploaded the correct test file. Missing: {missing[:5]}...")
        
    else:
        st.success(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
        
        # Prepare Data
        X_test = df[feature_cols]
        y_test = (df[target_col] == PHISHING_LABEL).astype(int)
        
        # Evaluate
        with st.spinner(f"Evaluating {selected_model_name}..."):
            pipeline = load_model(selected_model_name)
            y_pred = pipeline.predict(X_test)
            
            # Robust AUC Score calculation (Fixes SVM error)
            if hasattr(pipeline, "predict_proba"):
                y_score = pipeline.predict_proba(X_test)[:, 1]
            else:
                y_score = pipeline.decision_function(X_test)
            
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC Score": roc_auc_score(y_test, y_score),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "MCC Score": matthews_corrcoef(y_test, y_pred)
            }
        
        # Display Metrics
        st.subheader(f"Evaluation Metrics: {selected_model_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col2.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
        col3.metric("Precision", f"{metrics['Precision']:.4f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{metrics['Recall']:.4f}")
        col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        col6.metric("MCC Score", f"{metrics['MCC Score']:.4f}")
        
        st.divider()
        
        # Display Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(
            y_test, 
            y_pred, 
            labels=[0, 1],
            display_labels=["Legitimate", "Phishing"],
            cmap="Blues", 
            ax=ax
        )
                # --- MOVE LABELS TO TOP ---
        ax.xaxis.set_label_position('top') 
        ax.xaxis.tick_top()
        st.pyplot(fig)

else:
    st.info("Awaiting CSV file upload. Please upload the `test_data.csv` generated during training.")
