# Phishing Website Classification

## a. Problem Statement
## a. Problem Statement
Phishing remains one of the most common causes of data breaches, enabling credential theft, fraud, and ransomware access. Along with these cybersecurity threats, attackers trick users into revealing sensitive information by mimicking legitimate websites. Blacklist-based defenses fail against newly registered, short-lived domains. This project builds a classifier that predicts whether a website is **phishing** or **legitimate** from 30 extracted features covering URL structure, SSL status, domain age, and page behavior. Logistic Regression, Decision Tree, KNN, Naive Bayes, and SVM are compared on precision, recall, and MCC.

## b. Dataset Description
*   **Source:** [Phishing Website Dataset from Kaggle](https://www.kaggle.com/datasets/akashkr/phishing-website-dataset)
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

All observations below are based on the 2,211-sample test set, which contains **980 phishing** and **1,231 legitimate** websites. In this project, a **false negative** is a phishing website classified as legitimate (a missed threat). A **false positive** is a legitimate website wrongly flagged as phishing (a false alarm).

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Serves as a **strong and interpretable baseline**, with an Accuracy of 93.17% and an MCC of 0.861. The model is slightly conservative: its Precision (92.95%) is higher than its Recall (91.53%), so it prefers to avoid false alarms rather than catch every threat. It **missed 83 phishing websites** and **raised 68 false alarms**, which is the highest total error count among the four usable models (151 errors). Because it can only learn a linear decision boundary, it likely struggles with more complex interactions between website features. |
| **Decision Tree** | Improved on Logistic Regression mainly in **threat detection**, reducing missed phishing websites from 83 to **43** while keeping false alarms almost unchanged (67). This gives a Recall of **95.61%** and an F1 Score of 94.46%. The model's decisions rely heavily on two features: **`SSLfinal_State` (68.8%)** and **`URL_of_Anchor` (12.3%)**, which together account for roughly 81% of its importance. This means the model is only as reliable as these two features, and its performance may drop if they are unavailable or easily manipulated by attackers. |
| **K-Nearest Neighbors** | Achieved the **highest Precision of all models (97.80%)** and only **21 false alarms**, making it the best choice when wrongly blocking legitimate websites is costly. Its trade-off is a lower Recall (95.31%), with **46 phishing websites missed**. Because the website features take only a few discrete values, similar websites form tight groups, which suits a distance-based method like KNN. Overall Accuracy is **96.97%** and MCC is **0.939**. |
| **Naive Bayes** | The **weakest model** on this dataset (Accuracy 61.60%, MCC 0.407). It labels almost every website as phishing, giving a near-perfect Recall (**99.90%**, only 1 missed threat) but **848 false alarms** and a very low Precision (**53.59%**). In practice, this would frustrate users, since roughly two out of three safe websites would be blocked or flagged, and trust in the system would quickly erode. Despite its strong Recall, it is **not suitable for real-world use** in this form. Its AUC of **0.968** shows the model still ranks phishing sites above legitimate ones correctly, so the problem is not its ability to separate the classes but its **poorly calibrated decision threshold**. The likely cause is that Gaussian Naive Bayes assumes continuous, normally distributed inputs, while every feature here is discrete (-1, 0, 1). A categorical or Bernoulli variant would be better suited. |
| **Support Vector Machine** | Delivered the **best overall results** on this split: highest Accuracy (**97.15%**), F1 Score (**96.77%**) and MCC (**0.942**). It also has the best Recall of the four usable models (**96.43%**) while keeping Precision high (97.12%), with only **35 missed phishing websites** and **28 false alarms**. Its lead over KNN is very small (63 vs 67 total errors out of 2,211), so the two models are **effectively tied** and the difference should not be treated as conclusive. |

### Key Takeaways

- **Best overall balance:** Support Vector Machine, closely followed by K-Nearest Neighbors.
- **Fewest false alarms:** K-Nearest Neighbors (21).
- **Fewest missed phishing sites among the four usable models:** Support Vector Machine (35).
- **Not recommended without modification:** Least practical for real-world use - Naive Bayes, which is only suitable where missing a phishing website is unacceptable and a very high false alarm rate can be tolerated, such as a first-pass filter followed by manual review.

### Overall Winner for this Dataset: Support Vector Machine (SVM)
The Support Vector Machine is the best-performing model on this dataset, achieving the highest **Accuracy (97.15%)**, **AUC (0.9935)**, **F1 Score (96.77%)** and **MCC (0.9422)**, with the best Recall (96.43%) among the four usable models. K-Nearest Neighbors is a very close second and has the highest Precision (97.80%), so the gap between the two is small (63 vs 67 errors out of 2,211 test samples). SVM is selected as the winner for its better overall balance between false alarms and missed phishing websites.
