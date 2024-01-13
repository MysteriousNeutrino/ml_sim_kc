from __future__ import annotations
import numpy as np


def mse(y: np.ndarray) -> float:
    """Compute the mse impurity criterion for a given set of target values."""
    return np.mean((y - np.mean(y)) ** 2)


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mse criterion for a given set of target values."""
    num = mse(y_left) * y_left.size + mse(y_right) * y_right.size
    den = y_left.size + y_right.size
    return num / den


def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Find the best split for a node."""

    best_split = {}
    for feature in range(X.shape[1]):
        split_elements = X[:, feature]

        best_split_1_future = {}
        for i in split_elements:
            indexes_below_split_value = np.where(split_elements <= i)[0]
            indexes_above_split_value = np.where(split_elements > i)[0]

            y_left = np.array(y[indexes_below_split_value])
            y_right = np.array(y[indexes_above_split_value])

            mse = weighted_mse(y_left, y_right)
            best_split_1_future[i] = mse

        best_split[feature] = min(best_split_1_future, key=best_split_1_future.get)

    best_feature, best_thr = min(best_split.items(), key=lambda x: x[1])

    return best_feature, best_thr
