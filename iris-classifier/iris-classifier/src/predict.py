"""Predict the species of an iris flower from its measurements.

Usage:
    python -m src.predict 5.1 3.5 1.4 0.2
    (sepal length, sepal width, petal length, petal width, all in cm)
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris_model.joblib"
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run `python -m src.train` first.")

    values = [float(v) for v in sys.argv[1:]]
    bundle = joblib.load(MODEL_PATH)
    model, class_names = bundle["model"], bundle["class_names"]

    sample = pd.DataFrame([values], columns=FEATURES)
    pred = model.predict(sample)[0]
    probs = model.predict_proba(sample)[0]

    print(f"Prediction: {class_names[pred]}")
    for name, p in zip(class_names, probs):
        print(f"  {name:<12} {p:.1%}")


if __name__ == "__main__":
    main()
