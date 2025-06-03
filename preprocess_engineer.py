#!/usr/bin/env python3
"""
preprocess_and_feature_engineer.py

Loads raw data from SQLite, cleans missing values, encodes features, and saves
processed NumPy arrays for training.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import joblib

def load_data():
    conn = sqlite3.connect("churn.db")
    df = pd.read_sql("SELECT * FROM raw_customers", conn)
    conn.close()
    return df

def clean_and_encode(df: pd.DataFrame) -> (np.ndarray, np.ndarray, list):
    # Drop 'customerID'
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert 'TotalCharges' to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Binary-encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Identify categorical columns
    cat_cols = [col for col in df.columns
                if df[col].dtype == "object" and col != "Churn"]

    # Replace "No internet service" or "No phone service" with "No"
    for col in cat_cols:
        df[col] = df[col].replace({
            "No internet service": "No",
            "No phone service": "No"
        })

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Separate X and y
    y = df["Churn"].values
    X = df.drop(columns=["Churn"]).values
    feature_names = df.drop(columns=["Churn"]).columns.tolist()
    return X, y, feature_names

def save_processed_data(X, y):
    np.savez_compressed("processed_data.npz", X=X, y=y)
    print("Saved processed_data.npz (X & y)")

def main():
    df = load_data()
    print("Raw data shape:", df.shape)

    X, y, feature_names = clean_and_encode(df)
    print("After encoding shape:", X.shape)

    # Save arrays
    save_processed_data(X, y)

    # Save feature names for reference
    joblib.dump(feature_names, "feature_names.pkl")
    print("Saved feature_names.pkl")

if __name__ == "__main__":
    main()
