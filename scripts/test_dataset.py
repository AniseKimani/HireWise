import pandas as pd

file_path = "data/raw/upwork-jobs.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 rows:")
print(df.head())