# get_data.py

import kagglehub
import os
import shutil

# 1) Download the Telco‐Customer‐Churn dataset via Kagglehub
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
print("dataset_download returned:", path)

# 2) Ensure a local data/ folder exists
os.makedirs("data", exist_ok=True)

# 3) If `path` is a directory, look for the CSV inside it and copy
if os.path.isdir(path):
    csv_found = False
    for root, _, files in os.walk(path):
        for fname in files:
            if fname.lower().endswith(".csv"):
                src = os.path.join(root, fname)
                dst = os.path.join("data", "Telco-Customer-Churn.csv")
                print(f"Copying CSV from '{src}' -> '{dst}'")
                shutil.copy(src, dst)
                csv_found = True
                break
        if csv_found:
            break

    if not csv_found:
        raise FileNotFoundError(f"No .csv found under directory {path}")

# 4) If `path` looks like an archive (e.g., a .zip), unpack it into data/
else:
    try:
        shutil.unpack_archive(path, "data")
        print(f"Unpacked archive {path} into data/")
    except shutil.ReadError:
        raise RuntimeError(f"Returned path {path} is neither a folder nor a known archive format.")

    # After unpacking, find the CSV under data/ and move it to the root of data/
    csv_found = False
    for root, _, files in os.walk("data"):
        for fname in files:
            if fname.lower().endswith(".csv"):
                src = os.path.join(root, fname)
                dst = os.path.join("data", "Telco-Customer-Churn.csv")
                if src != dst:
                    print(f"Moving CSV from '{src}' -> '{dst}'")
                    shutil.move(src, dst)
                csv_found = True
                break
        if csv_found:
            break

    if not csv_found:
        raise FileNotFoundError("No .csv found in the unpacked archive under data/")

print("Telco-Customer-Churn.csv is now in the data/ folder.")
