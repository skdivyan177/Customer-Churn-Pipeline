#!/usr/bin/env python3
"""
ingest_and_store.py

Loads the raw Telco Customer Churn CSV into a local SQLite database.
"""

import sqlite3
import pandas as pd
import os

def main():
    csv_path = os.path.join("data", "Telco-Customer-Churn.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected CSV at {csv_path}")

    # Load with pandas
    df = pd.read_csv(csv_path)

    # Connect (or create) churn.db
    conn = sqlite3.connect("churn.db")

    # Write DataFrame to SQL table named 'raw_customers'
    df.to_sql("raw_customers", conn, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into churn.db → table raw_customers")

    conn.close()

if __name__ == "__main__":
    main()
