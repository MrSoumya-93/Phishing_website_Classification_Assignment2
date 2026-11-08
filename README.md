# Phishing Website Classification

**Author:** Soumyadeep Banerjee | **BITS ID:** 2026AJ05943
**Course:** Machine Learning (AIMLZG565), M.Tech (AIML), BITS Pilani WILP | **Assignment 2**

---

## a. Problem Statement

Phishing remains one of the most common causes of data breaches, enabling credential theft, fraud, and ransomware access. Along with these cybersecurity threats, attackers trick users into revealing sensitive information by mimicking legitimate websites. Blacklist-based defenses fail against newly registered, short-lived domains. This project builds a classifier that predicts whether a website is **phishing** or **legitimate** from 30 extracted features covering URL structure, SSL status, domain age, and page behavior. Logistic Regression, Decision Tree, KNN, Naive Bayes, and SVM are compared on precision, recall, and MCC.

---

## b. Dataset Description

| Property | Details |
| :--- | :--- |
| **Source** | [Phishing Website Dataset (Kaggle)](https://www.kaggle.com/datasets/akashkr/phishing-website-dataset) |
| **Instances** | 11,055 websites |
| **Features** | 30 discrete features (e.g., `SSLfinal_State`, `URL_of_Anchor`, `web_traffic`, `Prefix_Suffix`) |
| **Feature values** | Every feature takes one of three values: `-1`, `0`, or `1` |
| **Target variable** | `Result`: originally `-1` = Phishing and `1` = Legitimate. In this project it is recoded to `1` = Phishing and `0` = Legitimate, so Phishing is the positive class for all metrics |
| **Task type** | Binary classification |
| **Missing values** | None |
| **Train / Test split** | 80% / 20%, stratified, `random_state=42` (8,844 train / 2,211 test) |
| **Test set class balance** | 980 phishing (44.3%) and 1,231 legitimate (55.7%) |

The dataset exceeds the assignment minimums (at least 12 features and 500 instances). The 30 features describe the URL itself (length, IP address, special symbols, sub-domains), the security and domain profile (SSL state, domain age, DNS record, registration length), the HTML and JavaScript behavior of the page (anchors, form handlers, pop-ups, iframes), and traffic or reputation signals (web traffic, page rank, Google index, links pointing to the page).

---

## c. GitHub Repository Link

**Repository:** [https://github.com/MrSoumya-93/Phishing_website_Classification_Assignment2](https://github.com/MrSoumya-93/Phishing_website_Classification_Assignment2)

**Live Streamlit App:** [https://phishingwebsiteclassification.streamlit.app/](https://phishingwebsiteclassification.streamlit.app/)

---

## d. Models Used & Evaluation Metrics

All five models were trained on the same 80/20 stratified split (`random_state=42`) and evaluated on the same 20% test set of 2,211 instances. Phishing is the positive class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9317 | 0.9786 | 0.9295 | 0.9153 | 0.9224 | 0.8615 |
| Decision Tree | 0.9502 | 0.9861 | 0.9333 | 0.9561 | 0.9446 | 0.8997 |
| K-Nearest Neighbors | 0.9697 | 0.9931 | **0.9780** | 0.9531 | 0.9654 | 0.9387 |
| Naive Bayes | 0.8621 | 0.9677 | 0.7796 | 0.9602 | 0.8605 | 0.7424 |
| Support Vector Machine | **0.9715** | **0.9935** | 0.9712 | **0.9643** | **0.9677** | **0.9422** |

*Bold values mark the best score in each column.*

### Confusion Matrices (Test Set, 2,211 instances)

| ML Model Name | True Negative | False Positive | False Negative | True Positive | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 1163 | 68 | 83 | 897 | 151 |
| Decision Tree | 1164 | 67 | 43 | 937 | 110 |
| K-Nearest Neighbors | 1210 | 21 | 46 | 934 | 67 |
| Naive Bayes | 965 | 266 | 39 | 941 | 305 |
| Support Vector Machine | 1203 | 28 | 35 | 945 | 63 |

**Terminology:** A **false negative (FN)** is a phishing website classified as legitimate (a missed threat). A **false positive (FP)** is a legitimate website wrongly flagged as phishing (a false alarm).

### Model Configuration Notes
- **Naive Bayes:** Gaussian Naive Bayes with `var_smoothing=1e-3` (default `1e-9`), chosen to balance Recall against Accuracy, Precision, and MCC (see the Naive Bayes tuning note below).
- **K-Nearest Neighbors:** `n_neighbors=10`.
- **Support Vector Machine:** RBF kernel with `C=20`.

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Serves as a **strong and interpretable baseline**, with an Accuracy of 93.17% and an MCC of 0.861. The model is slightly conservative: its Precision (92.95%) is higher than its Recall (91.53%), so it prefers to avoid false alarms rather than catch every threat. It **missed 83 phishing websites** (the most of any model) and **raised 68 false alarms**, for 151 errors in total. Because it can only learn a linear decision boundary, it likely struggles with more complex interactions between website features. |
| **Decision Tree** | Improved on Logistic Regression mainly in **threat detection**, reducing missed phishing websites from 83 to **43** while keeping false alarms almost unchanged (67). This gives a Recall of **95.61%** and an F1 Score of 94.46%. The model's decisions rely heavily on two features: **`SSLfinal_State` (68.8%)** and **`URL_of_Anchor` (12.3%)**, which together account for roughly 81% of its importance. This means the model is only as reliable as these two features, and its performance may drop if they are unavailable or easily manipulated by attackers. |
| **K-Nearest Neighbors** | Achieved the **highest Precision of all models (97.80%)** and only **21 false alarms**, making it the best choice when wrongly blocking legitimate websites is costly. Its trade-off is a lower Recall (95.31%), with **46 phishing websites missed**. Because the website features take only a few discrete values, similar websites form tight groups, which suits a distance-based method like KNN. Overall Accuracy is **96.97%** and MCC is **0.939**. |
| **Naive Bayes** | A **recall-oriented model** and the **lowest overall performer** (Accuracy 86.21%, MCC 0.742). It catches **941 of 980 phishing websites** (Recall **96.02%**) and misses only **39**, but it is the most aggressive model: it raised **266 false alarms**, wrongly flagging about **22% of legitimate websites (roughly one in five)** as phishing, which lowers Precision to **77.96%**. In practice, this would frustrate users, because many safe websites would be blocked or flagged. Its AUC of **0.968** shows that the model ranks phishing sites above legitimate ones correctly, so the weakness lies in its **decision threshold** rather than in its ability to separate the classes. Gaussian Naive Bayes assumes continuous inputs, while every feature here is discrete (-1, 0, 1), which makes it over-confident in predicting phishing. |
| **Support Vector Machine** | Delivered the **best overall results** on this split: highest Accuracy (**97.15%**), AUC (**0.9935**), F1 Score (**96.77%**) and MCC (**0.942**). It also has the **highest Recall of all five models (96.43%)** while keeping Precision high (97.12%), with only **35 missed phishing websites** and **28 false alarms**. Its lead over KNN is very small (63 vs 67 total errors out of 2,211), so the two models are **effectively tied** and the difference should not be treated as conclusive. |

### Note on Naive Bayes Tuning (`var_smoothing`)

For Naive Bayes, `var_smoothing` controls a clear **trade-off between Recall and the other metrics**:

- With the default value (`1e-9`), the model reached a near-perfect **Recall (99.90%)**, but at the cost of **848 false alarms**, with Accuracy falling to **61.60%**, Precision to **53.59%**, and MCC to **0.407**.
- Increasing `var_smoothing` improves **Accuracy, Precision, and MCC**, but **Recall decreases**, because the model becomes less eager to label websites as phishing.
- In other words, **focusing on Accuracy, Precision, and MCC lowers Recall, while focusing on Recall lowers Accuracy, Precision, and MCC**.

To strike a balance, **`var_smoothing=1e-3`** was used. This keeps Recall high (**96.02%**) while cutting false alarms from **848 to 266** and raising Accuracy from **61.60% to 86.21%**.

### Key Takeaways

- **Best overall balance:** Support Vector Machine, closely followed by K-Nearest Neighbors.
- **Fewest false alarms:** K-Nearest Neighbors (21), followed by Support Vector Machine (28).
- **Fewest missed phishing websites:** Support Vector Machine (35), followed by Naive Bayes (39) and Decision Tree (43).
- **Naive Bayes:** After tuning, it is a high-Recall model (96.02%) but with the most false alarms (266) and the lowest Accuracy (86.21%). It is best suited to a first-pass filter followed by manual review, rather than as a standalone detector.
- **Most dependent on a single signal:** Decision Tree, where `SSLfinal_State` alone carries 68.8% of the model's importance.

### Overall Winner for this Dataset: Support Vector Machine (SVM)

The **Support Vector Machine** is the best-performing model on this dataset, achieving the highest **Accuracy (97.15%)**, **AUC (0.9935)**, **F1 Score (96.77%)**, **MCC (0.9422)**, and the highest **Recall (96.43%)** of all five models. K-Nearest Neighbors is a very close second and has the highest Precision (97.80%), so the gap between the two is small (63 vs 67 errors out of 2,211 test samples). SVM is selected as the winner for its better overall balance between false alarms and missed phishing websites.

---

## e. Streamlit Application

**Live App:** [https://phishingwebsiteclassification.streamlit.app/](https://phishingwebsiteclassification.streamlit.app/)

The app loads the saved models from the `model/` folder and **does not retrain** on uploaded data. It provides:

1. **Dataset upload:** upload a test CSV containing the 30 feature columns and the `Result` target column (`test_data.csv` in this repository can be used).
2. **Model selection:** a dropdown to choose any of the five trained models.
3. **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 Score, and MCC for the selected model.
4. **Confusion matrix:** the matrix of predicted versus actual classes on the uploaded data.

---

## f. Repository Structure

```
Phishing_website_Classification_Assignment2/
│-- app.py                                  # Streamlit application
│-- requirements.txt                        # Python dependencies
│-- README.md                               # Project documentation
│-- test_data.csv                           # Held-out test data (2,211 rows, features + target)
└-- model/
    │-- dataset.csv                         # Full dataset (11,055 rows)
    │-- data_loader.py                      # Loads data and separates features from target
    │-- splitting.py                        # Stratified 80/20 split and test-data export
    │-- evaluation.py                       # Shared metric computation and saving
    │-- logistic_regression.py              # Model 1 training script
    │-- decision_tree.py                    # Model 2 training script
    │-- knn.py                              # Model 3 training script
    │-- naive_bayes.py                      # Model 4 training script
    │-- svm.py                              # Model 5 training script
    │-- logistic_regression.joblib          # Saved Model 1
    │-- decision_tree.joblib                # Saved Model 2
    │-- knn.joblib                          # Saved Model 3
    │-- naive_bayes.joblib                  # Saved Model 4
    │-- svm.joblib                          # Saved Model 5
    │-- feature_schema.json                 # Feature names and target encoding used by the app
    │-- k-nearest-neighbors_metrics.json    # Saved evaluation metrics
    └-- support_vector_machine_metrics.json # Saved evaluation metrics
```
