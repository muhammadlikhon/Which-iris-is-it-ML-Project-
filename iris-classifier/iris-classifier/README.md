# Iris Classifier

A small, end-to-end machine learning project: classify iris flowers into three
species from four measurements, using scikit-learn.

## What it demonstrates
- Loading a dataset and making a stratified train/test split
- Wrapping preprocessing and model in a `Pipeline` (no data leakage)
- Comparing two models with 5-fold cross-validation
- Evaluating the winner **once** on a held-out test set
- Saving the trained model and using it for predictions

## Project layout
```
iris-classifier/
├── src/
│   ├── data.py       # load + split data
│   ├── train.py      # compare models, evaluate, save model + plot
│   └── predict.py    # command-line predictions
├── models/           # trained model (created by train.py)
├── outputs/          # confusion matrix plot (created by train.py)
└── requirements.txt
```

## Quick start
```bash
pip install -r requirements.txt

python -m src.train
python -m src.predict 5.9 3.0 5.1 1.8   # sepal L/W, petal L/W in cm
```

Example output:
```
Prediction: virginica
  setosa       0.4%
  versicolor   31.4%
  virginica    68.2%
```

## Results
Test accuracy is about 93% on 30 held-out flowers. Setosa is separated
perfectly; the small number of errors are versicolor/virginica mix-ups, which
overlap in feature space.

## Ideas to extend
- Add `GridSearchCV` to tune hyperparameters
- Try other models (SVM, k-NN, gradient boosting)
- Swap in your own CSV in `src/data.py`
- Wrap `predict.py` in a small FastAPI or Streamlit app
