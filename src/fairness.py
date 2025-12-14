import joblib
import numpy as np
import pandas as pd

from fairlearn.metrics import MetricFrame, selection_rate, false_negative_rate
from sklearn.model_selection import train_test_split

MODEL_PATH = "artifacts/model.joblib"
DATA_PATH = "data/data.csv"

def load_df():
    df = pd.read_csv(DATA_PATH)
    if "gender" in df.columns:
        df["gender"] = pd.factorize(df["gender"])[0]
    for col in ["sno", "SNo", "index"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.dropna()
    return df

def main():
    model = joblib.load(MODEL_PATH)
    df = load_df()

    X = df.drop("target", axis=1)
    y = df["target"].map({"yes": 1, "no": 0}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    # bucket age
    age = X_test["age"]
    age_bins = pd.cut(age, bins=list(range(0, 121, 20)), right=False, include_lowest=True)

    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "false_negative_rate": false_negative_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=age_bins
    )

    print("By age bin:\n", mf.by_group)
    print("\nOverall:\n", mf.overall)
    print("\nGroup disparities:")
    print("selection_rate range:", mf.by_group["selection_rate"].max() - mf.by_group["selection_rate"].min())
    print("false_negative_rate range:", mf.by_group["false_negative_rate"].max() - mf.by_group["false_negative_rate"].min())

if __name__ == "__main__":
    main()
