import pandas as pd
import os

csv_path = r"u:\FGMFN_S-A-A\data\processed\train.csv"
if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
else:
    try:
        df = pd.read_csv(csv_path)
        print("--- HEAD ---")
        print(df.head())
        print("\n--- LABEL COUNTS ---")
        if 'label' in df.columns:
            print(df['label'].value_counts())
        else:
            print("No 'label' column found!")
            print("Columns:", df.columns)
    except Exception as e:
        print(f"Error reading SCV: {e}")
