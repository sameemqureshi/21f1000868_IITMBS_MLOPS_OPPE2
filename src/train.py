import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
import os
os.makedirs("artifacts", exist_ok=True)

DATA_PATH = "data/data.csv"
MODEL_PATH = "artifacts/model.joblib"
FEAT_PATH = "artifacts/feature_columns.json"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # match notebook intent: factorize gender if present
    if "gender" in df.columns:
        df["gender"] = pd.factorize(df["gender"])[0]

    # drop id-like columns if present
    for col in ["sno", "SNo", "index"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # notebook drops NA
    df = df.dropna()

    return df

def build_pipeline(X: pd.DataFrame):
    # detect categorical vs numeric
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(max_iter=2000)

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    return pipe, num_cols, cat_cols

def main():
    np.random.seed(42)
    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"].map({"yes": 1, "no": 0}).astype(int)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe, num_cols, cat_cols = build_pipeline(X_train)

    grid = {
        "clf__C": np.logspace(-4, 4, 20),
        "clf__solver": ["liblinear"],
    }

    search = RandomizedSearchCV(
        pipe, param_distributions=grid, cv=5, n_iter=20, verbose=1, n_jobs=-1
    )

    search.fit(X_train, y_train)

    proba = search.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("Best params:", search.best_params_)
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, pred))

    joblib.dump(search.best_estimator_, MODEL_PATH)
    with open(FEAT_PATH, "w") as f:
        json.dump({"raw_features": list(X.columns)}, f, indent=2)

if __name__ == "__main__":
    main()
