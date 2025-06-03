#!/usr/bin/env python3
"""
train_and_evaluate.py

Loads processed data (X, y), performs train/test split, trains models,
evaluates performance, and saves best model pipeline.
"""

import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report
)
import os

def load_processed_data():
    data = np.load("processed_data.npz")
    return data["X"], data["y"]

def baseline_logistic(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Logistic Regression Results:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
    return clf

def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best RF params:", grid.best_params_)
    return grid.best_estimator_

def main():
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 1) Baseline
    log_model = baseline_logistic(X_train, X_test, y_train, y_test)

    # 2) Hyperparameter-tuned Random Forest
    best_rf = tune_random_forest(X_train, y_train)
    y_pred_rf = best_rf.predict(X_test)
    print("Random Forest Results:")
    print(classification_report(y_test, y_pred_rf))
    print("RF ROC AUC:", roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1]))

    # Save the best model for downstream pipeline usage
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_rf, "model/best_rf_model.pkl")
    print("Saved best_rf_model.pkl in model/")

if __name__ == "__main__":
    main()
