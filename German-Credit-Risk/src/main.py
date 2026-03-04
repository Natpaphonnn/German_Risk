import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """โหลด CSV และ drop unnamed index column."""
    df = pd.read_csv(path)
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """แปลง categorical เป็นตัวเลข, จัดการ missing values, แยก X/y."""
    df = df.copy()

    # เติม missing values ใน Saving accounts / Checking account ด้วย "unknown"
    df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
    df["Checking account"] = df["Checking account"].fillna("unknown")

    # Encode target: good=0, bad=1
    df["Risk"] = df["Risk"].map({"good": 0, "bad": 1})

    # Label-encode categorical columns
    cat_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    y = df["Risk"]
    X = df.drop(columns=["Risk"])

    return X, y, encoders


# ── Training ──────────────────────────────────────────────────────────────────

def train_models(X_train, y_train):
    """เทรน 3 โมเดลแล้วคืนเป็น dict."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"[OK] {name} trained")
    return models


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(models: dict, X_test, y_test):
    """แสดงผล accuracy, AUC, classification report ของแต่ละโมเดล."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results[name] = {"accuracy": acc, "auc": auc}

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["good", "bad"]))
    return results


def plot_confusion_matrices(models: dict, X_test, y_test, save_path: str = None):
    """พล็อต confusion matrix ของทุกโมเดลในรูปเดียว."""
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["good", "bad"], yticklabels=["good", "bad"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[OK] Confusion matrix saved to {save_path}")
    plt.show()


def plot_feature_importance(model, feature_names, save_path: str = None):
    """พล็อต feature importance ของ tree-based model."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], orient="h")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[OK] Feature importance saved to {save_path}")
    plt.show()


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    # 1) Load
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "german_credit_data.csv")
    df = load_data(data_path)
    print(f"Dataset shape: {df.shape}")

    # 2) Preprocess
    X, y, encoders = preprocess(df)

    # 3) Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # 5) Train
    models = train_models(X_train, y_train)

    # 6) Evaluate
    results = evaluate(models, X_test, y_test)

    # 7) Plots
    plot_confusion_matrices(models, X_test, y_test)

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = models[best_name]
    print(f"\nBest model by AUC: {best_name} ({results[best_name]['auc']:.4f})")

    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, X.columns.tolist())

    # 8) Save best model
    output_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump({"model": best_model, "scaler": scaler, "encoders": encoders}, model_path)
    print(f"[OK] Best model saved to {model_path}")


if __name__ == "__main__":
    main()
