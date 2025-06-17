# Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using various machine learning models and data balancing strategies. It is based on the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and inspired by the notebook by Janio Bachmann, extended with deeper experimentation on class imbalance techniques.

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Imbalance**: Only 492 frauds (~0.172%)

All features except `Time` and `Amount` are anonymized via PCA transformation. The target column is `Class` (1 = Fraud, 0 = Not Fraud).

---

## Objective

The goal is to:
- Evaluate the impact of different data balancing techniques on fraud detection
- Compare the performance of five machine learning models
- Use precision, recall, F1-score, and confusion matrix as evaluation metrics

---

## ⚖Balancing Techniques

This project compares five strategies to handle class imbalance:

1. **RandomUnderSampler**
2. **NearMiss**
3. **SMOTE**
4. **SMOTE + RandomUnderSampler**
5. **SMOTE + NearMiss**

---

## Models Evaluated

- Logistic Regression
- K-Nearest Neighbors (K-NN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

---

## 📊 Evaluation Metrics

Each model is evaluated using the following metrics on an imbalanced test set:
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📈 Results Summary

- **Random Forest** consistently achieved the best F1-Score, especially when used with `SMOTE` or `SMOTE + RandomUnderSampler`.
- **Undersampling-only** strategies performed poorly, showing low precision despite high recall.
- Combining **SMOTE with RandomUnderSampler** offers the best balance between precision and recall.
