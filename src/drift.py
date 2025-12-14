import json
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

DATA_PATH = "data/data.csv"
PRED_PATH = "artifacts/random_100.json"

def load_train():
    df = pd.read_csv(DATA_PATH)
    if "gender" in df.columns:
        df["gender"] = pd.factorize(df["gender"])[0]
    for col in ["sno", "SNo", "index"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.dropna()
    return df.drop("target", axis=1)

def load_pred():
    with open(PRED_PATH) as f:
        payload = json.load(f)[0]
    return pd.DataFrame(payload["rows"])

def main():
    ref = load_train()
    cur = load_pred()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    out_html = "artifacts/drift_report.html"
    report.save_html(out_html)
    print("Saved:", out_html)

if __name__ == "__main__":
    main()
