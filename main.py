import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = Path("outputs")


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_regression():
    print_header("PART 1: LINEAR REGRESSION - California House Prices")

    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    features = housing.feature_names

    df = pd.DataFrame(X, columns=features)
    df["target"] = y
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target: median house value in $100,000s")
    print(df.describe().T[["mean", "std", "min", "max"]].round(2))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    print("\nModel coefficients:")
    for name, coef in zip(features, model.coef_):
        print(f"  {name:<20} {coef: .4f}")

    print("\nEvaluation metrics (test set):")
    print(f"  MSE  (Mean Squared Error)     : {mse:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  R-squared                     : {r2:.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
    limit = [0, 5.5]
    ax.plot(limit, limit, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlim(limit)
    ax.set_ylim(limit)
    ax.set_xlabel("Actual price ($100,000s)")
    ax.set_ylabel("Predicted price ($100,000s)")
    ax.set_title(f"Linear Regression\nMSE={mse:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regression_predictions.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved plot: {OUTPUT_DIR / 'regression_predictions.png'}")


def run_classification():
    print_header("PART 2: DECISION TREE CLASSIFIER - Breast Cancer Diagnosis")

    data = load_breast_cancer()
    X = data.data
    y = data.target
    labels = data.target_names

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {labels[0]} (0), {labels[1]} (1)")
    print(f"Class distribution: malignant={int((y == 0).sum())}, benign={int((y == 1).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    print("\nEffect of max_depth (train vs test accuracy):")
    for depth in [2, 3, 4, 6, None]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        tree.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, tree.predict(X_train))
        test_acc = accuracy_score(y_test, tree.predict(X_test))
        depth_label = str(depth) if depth is not None else "unlimited"
        print(f"  max_depth={depth_label:<9} train={train_acc:.4f}  test={test_acc:.4f}")

    final_depth = 4
    model = DecisionTreeClassifier(max_depth=final_depth, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nFinal model: DecisionTreeClassifier(max_depth={final_depth})")
    print(f"Accuracy: {accuracy:.4f}")

    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"                Predicted malignant  Predicted benign")
    print(f"Actual malignant           {cm[0, 0]:>5}            {cm[0, 1]:>5}")
    print(f"Actual benign              {cm[1, 0]:>5}            {cm[1, 1]:>5}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=labels, digits=4))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Decision Tree Confusion Matrix (Accuracy={accuracy:.3f})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=120)
    plt.close()
    print(f"Saved plot: {OUTPUT_DIR / 'confusion_matrix.png'}")

    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(
        model,
        feature_names=data.feature_names,
        class_names=labels.tolist(),
        filled=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Decision Tree (max_depth={final_depth})")
    fig.savefig(OUTPUT_DIR / "decision_tree.png", dpi=120)
    plt.close(fig)
    print(f"Saved plot: {OUTPUT_DIR / 'decision_tree.png'}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_regression()
    run_classification()
    print_header("SUMMARY")
    print("Supervised ML pipeline: load data -> train/test split -> fit -> predict -> evaluate")
    print("Regression metrics: MSE, RMSE, R-squared")
    print("Classification metrics: Accuracy, Confusion Matrix, Precision/Recall/F1")


if __name__ == "__main__":
    main()
