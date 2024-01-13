import numpy as np


def split(X: np.ndarray, y: np.ndarray, feature: int) -> float:
    """Find the best split for a node (one feature)"""

    split_elements = X[:, feature]

    best_split = {}
    for i in split_elements:
        indexes_below_split_value = np.where(split_elements <= i)[0]
        indexes_above_split_value = np.where(split_elements > i)[0]

        y_left = np.array(y[indexes_below_split_value])
        y_right = np.array(y[indexes_above_split_value])

        mse = weighted_mse(y_left, y_right)
        best_split[i] = mse

    max_key = min(best_split, key=best_split.get)

    return max_key


def mse(y: np.ndarray) -> np.ndarray:
    """Compute the mean squared error of a vector."""

    y_pred = np.mean(y)

    return np.mean((y - y_pred) ** 2)


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    y_left_size = y_left.size
    y_right_size = y_right.size
    return (mse(y_left) * y_left_size + mse(y_right) * y_right_size) / (y_right_size + y_left_size)
