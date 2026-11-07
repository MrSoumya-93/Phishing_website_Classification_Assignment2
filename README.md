# Phishing Website Classification

## a. Problem Statement
Phishing attacks are one of the most common cybersecurity threats, where attackers trick users into revealing sensitive information by mimicking legitimate websites. The objective of this project is to build and evaluate multiple machine learning classification models to automatically detect and classify a website as either **Legitimate** or **Phishing** based on 30 extracted URL and website features. 

## b. Dataset Description
*   **Source:** UCI Machine Learning Repository (Phishing Websites Dataset)
*   **Instances:** 11,055 rows
*   **Features:** 30 integer/ternary features (e.g., `SSLfinal_State`, `URL_of_Anchor`, `web_traffic`, `Prefix_Suffix`).
*   **Target Variable:** `Result` (Originally encoded as `-1` for Phishing and `1` for Legitimate. In this pipeline, it is mapped to `1` for Phishing and `0` for Legitimate to make Phishing the positive class for evaluation metrics).

## c. GitHub Repository Link
**Repository:** [https://github.com/MrSoumya-93/Phishing_website_Classification_Assignment2](https://github.com/MrSoumya-93/Phishing_website_Classification_Assignment2)

## d. Models Used & Evaluation Metrics
All 5 models were trained using an 80/20 stratified train-test split (Random State = 42). The evaluation metrics below are calculated on the 20% test set (2,211 instances).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9317 | 0.9786 | 0.9295 | 0.9153 | 0.9224 | 0.8615 |
| Decision Tree | 0.9502 | 0.9861 | 0.9333 | 0.9561 | 0.9446 | 0.8997 |
| K-Nearest Neighbors | 0.9697 | 0.9931 | 0.9780 | 0.9531 | 0.9654 | 0.9387 |
| Naive Bayes | 0.6160 | 0.9677 | 0.5359 | 0.9990 | 0.6975 | 0.4067 |
| Support Vector Machine | 0.9715 | 0.9935 | 0.9712 | 0.9643 | 0.9677 | 0.9422 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | A strong baseline model (MCC 0.861). It performed slightly conservatively (Precision > Recall), missing 83 phishing URLs while raising 68 false alarms. It assumes a linear decision boundary, which works well but misses complex non-linear feature interactions. |
| **Decision Tree** | Outperformed LR by drastically reducing false negatives (missing only 43 phishing URLs). Feature importance analysis revealed that the tree relies heavily on `SSLfinal_State` (68.8%) and `URL_of_Anchor` (12.3%), meaning its high accuracy is tied to the reliability of these specific features. |
| **K-Nearest Neighbors** | Tuned with `n_neighbors=10`. Highly reliable with the highest precision (97.8%), reducing false alarms to just 21. It excelled because the 30 binary/ternary features create distinct, localized clusters in the multi-dimensional feature space that distance metrics can easily separate. |
| **Naive Bayes** | The worst performer (MCC 0.407). The algorithm's core assumption that all features are independent fails completely here (URL features are highly correlated). Consequently, it became overly confident in predicting the Phishing class, resulting in massive false positives (848) and terrible Precision (53.6%). |
| **Support Vector Machine** | **Overall Winner.** Tuned with an RBF kernel and `C=20`. Achieved the highest Accuracy (97.15%), AUC (99.35%), F1 (96.77%), and MCC (94.22%). By drawing complex hyperplanes, it perfectly balanced Precision (97.1%) and Recall (96.4%), missing only 35 phishing URLs while raising 28 false alarms. It is the most robust model for production. |

### Overall Winner for this Dataset: Support Vector Machine (SVM)
The Support Vector Machine is the definitive winner for this dataset. It achieves the highest score across all six evaluation metrics. Its ability to handle non-linear relationships via the RBF kernel allows it to outperform simpler models while maintaining exceptional stability (evidenced by its near-perfect AUC and 94.22% MCC score).
