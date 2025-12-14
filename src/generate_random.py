import pandas as pd
import numpy as np

DATA_PATH = "data/data.csv"
OUT_PATH = "artifacts/random_100.json"

def main():
    df = pd.read_csv(DATA_PATH)

    if "gender" in df.columns:
        df["gender"] = pd.factorize(df["gender"])[0]
    for col in ["sno", "SNo", "index"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.dropna()
    X = df.drop("target", axis=1)

    sample = X.sample(n=100, replace=True, random_state=42).reset_index(drop=True)
    payload = {"rows": sample.to_dict(orient="records")}
    pd.Series([payload]).to_json(OUT_PATH, orient="records", indent=2)
    print(f"Wrote {OUT_PATH}")

if __name__ == "__main__":
    main()
