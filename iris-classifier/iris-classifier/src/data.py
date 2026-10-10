"""Data loading and splitting."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_data():
    """Return features (DataFrame), target (Series) and class names."""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y, list(iris.target_names)


def split_data(X, y, test_size=0.2):
    """Stratified train/test split so each class is represented equally."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
