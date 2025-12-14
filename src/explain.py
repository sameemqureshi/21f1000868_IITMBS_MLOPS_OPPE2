import joblib
import numpy as np
import pandas as pd
import shap
import os
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

MODEL_PATH = "artifacts/model.joblib"
DATA_PATH = "data/data.csv"
os.makedirs("artifacts", exist_ok=True)

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
    pred = (proba >= 0.5).astype(int)

    fn_mask = (y_test.values == 1) & (pred == 0)
    X_fn = X_test.loc[fn_mask]

    if len(X_fn) == 0:
        print("No false negatives in this split; try different seed or threshold.")
        return

    # SHAP for linear-ish models: explain in feature space AFTER preprocessing
    # easiest: use shap.Explainer(model.predict_proba, X_train) for pipeline
    explainer = shap.Explainer(model.predict_proba, X_train)
    shap_values = explainer(X_fn)
    

    # ---- SHAP visualizations (saved to artifacts/) ----

    # Summary plot for FALSE NEGATIVES (impact for class=1)
    sv = shap_values.values[:, :, 1]  # contributions to class 1
    plt.figure()
    shap.summary_plot(
        sv,
        features=X_fn,
        feature_names=shap_values.feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("artifacts/shap_summary_false_negatives.png", dpi=200)
    plt.close()

    # Bar plot (global importance for these false negatives)
    mean_abs = np.mean(np.abs(sv), axis=0)
    order = np.argsort(mean_abs)[::-1]

    plt.figure()
    shap.summary_plot(
        sv[:, order],
        features=X_fn.iloc[:, order] if hasattr(X_fn, "iloc") else X_fn,
        feature_names=np.array(shap_values.feature_names)[order],
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig("artifacts/shap_bar_false_negatives.png", dpi=200)
    plt.close()

    print("Saved SHAP plots to:")
    print("- artifacts/shap_summary_false_negatives.png")
    print("- artifacts/shap_bar_false_negatives.png")

    # Focus on contribution to class 1 probability:
    # shap_values.values shape: (n_samples, n_features, n_classes)
    sv = shap_values.values[:, :, 1]
    mean_abs = np.mean(np.abs(sv), axis=0)

    feat_names = shap_values.feature_names
    top_idx = np.argsort(mean_abs)[::-1][:8]

    print("\nTop drivers (mean |SHAP|) for FALSE NEGATIVES (true=1, pred=0):")
    for i in top_idx:
        print(f"- {feat_names[i]}: {mean_abs[i]:.4f}")

    # Plain-English summary pattern:
    print("\nPlain-English summary:")
    print(
        "False negatives are the patients where the model predicted 'no disease' "
        "even though they truly had disease. The model’s decision for these cases "
        "is most sensitive to the features listed above (largest average SHAP impact). "
        "Typically, these features are pushing the predicted probability DOWN for class 1 "
        "(heart disease), causing the miss."
    )

if __name__ == "__main__":
    main()
