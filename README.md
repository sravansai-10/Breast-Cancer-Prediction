# Breast-Cancer-Prediction
End‑to‑end ML pipeline for the WDBC breast‑cancer dataset: loads raw JSON/.data, scales features, tunes a RandomForest (≈ 99 % ROC‑AUC), exports model + confusion/ROC plots, and offers SHAP feature insights.
<!-- Title & Badges -->
# 🩺 Breast‑Cancer Prediction │ Wisconsin Diagnostic Dataset  
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predict malignant vs benign tumours with a **99 % ROC‑AUC** Random‑Forest model, visual interpretability, and a one‑click notebook demo.


## Project Highlights
End‑to‑end workflow         –    data loading, stratified splits, cross‑validated hyper‑parameter tuning, evaluation, and model persistence.
Scikit‑learn Pipeline       – `  StandardScaler → RandomForestClassifier`, tuned with `GridSearchCV`.
Evaluation assets           –    confusion‑matrix and ROC‑curve images saved to `artifacts/`.
Model interpretability      –    optional SHAP summary plot of feature importance.
Ready for reuse             –    standalone script (`breast_cancer_prediction.py`) and interactive notebook.


Results (held‑out test set)
Metric	Score
Accuracy	0.96
ROC‑AUC	0.989

Repository Structure
Copy
Edit
├─ Breast‑Cancer‑Prediction.ipynb
├─ requirements.txt
├─ artifacts/
│    ├─ roc_curve.png
│    ├─ confusion_matrix.png
│    ├─ shap_summary.png
│    └─ best_model.joblib
└─ LICENSE
Dataset
Breast‑Cancer Wisconsin Diagnostic (WDBC) – 569 samples, 30 numeric features. Source: UCI Machine Learning Repository.
MIT License
Copyright © 2025 Sravan Sai Methuku
Released under the MIT License – see LICENSE for details.

