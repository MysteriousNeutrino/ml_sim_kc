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


def split(X: np.ndarray, y: np.ndarray, feature: int) -> float:
    """Find the best split for a node."""
    node_size = y.size
    if node_size < 2:
        return None, None

    node_mse = mse(y)
    best_mse = node_mse
    best_thr = None

    thresholds = np.unique(X[:, feature])
    for thr in thresholds:
        left = y[X[:, feature] <= thr]
        right = y[X[:, feature] > thr]

        if left.size == 0 or right.size == 0:
            continue

        weihted_mse = weighted_mse(left, right)
        if weihted_mse < best_mse:
            best_mse = weihted_mse
            best_thr = thr

    return best_thr
