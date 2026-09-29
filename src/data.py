import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_and_split_data():
    wine = load_wine()

    # Validate feature count
    if wine.data.shape[1] != 13:
        raise ValueError("Wine dataset must have exactly 13 features.")

    # Validate null values
    if np.isnan(wine.data).any():
        raise ValueError("Wine dataset contains null values.")

    if np.isnan(wine.target).any():
        raise ValueError("Wine target contains null values.")

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=0.2,
        random_state=42,
        stratify=wine.target,
    )

    return X_train, X_test, y_train, y_test
