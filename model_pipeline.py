#!/usr/bin/env python3
"""
model_pipeline.py

Creates a full scikit-learn Pipeline (scaling + classifier), trains on processed data,
and serializes the entire pipeline to disk (for easy deployment).
"""

import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from train_and_evaluate import load_processed_data  # reuse the data-loading function

def main():
    X, y = load_processed_data()

    # Train/test split (80/20) again
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    y_pred = pipeline.predict(X_test)
    print("Pipeline results:")
    print(classification_report(y_test, y_pred))
    print("Pipeline ROC AUC:", roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1]))

    # Serialize pipeline
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/retention_pipeline.pkl")
    print("Saved retention_pipeline.pkl in model/")

if __name__ == "__main__":
    main()
