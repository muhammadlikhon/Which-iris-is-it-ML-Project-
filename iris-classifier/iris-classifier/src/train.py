"""Train, compare, evaluate and save a classifier.

Usage:
    python -m src.train
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import RANDOM_STATE, load_data, split_data

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "iris_model.joblib"
PLOT_PATH = ROOT / "outputs" / "confusion_matrix.png"


def build_candidates():
    """Candidate models, each wrapped in a pipeline with scaling."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        ),
    }


def main():
    X, y, class_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

    # 1. Compare models with 5-fold cross-validation on the training set only
    print("Cross-validation accuracy (5-fold):")
    scores = {}
    for name, model in build_candidates().items():
        cv = cross_val_score(model, X_train, y_train, cv=5)
        scores[name] = cv.mean()
        print(f"  {name:<20} {cv.mean():.3f} (+/- {cv.std():.3f})")

    # 2. Pick the best and fit it on the full training set
    best_name = max(scores, key=scores.get)
    best_model = build_candidates()[best_name]
    best_model.fit(X_train, y_train)
    print(f"\nBest model: {best_name}")

    # 3. Evaluate once on the held-out test set
    y_pred = best_model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 4. Save the confusion matrix and the model
    PLOT_PATH.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names, cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion matrix: {best_name}")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Saved plot  -> {PLOT_PATH.relative_to(ROOT)}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "class_names": class_names}, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
