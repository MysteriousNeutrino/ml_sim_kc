import numpy as np


def mse(y: np.ndarray) -> np.ndarray:
    """Compute the mean squared error of a vector."""

    y_pred = np.mean(y)

    return np.mean((y - y_pred) ** 2)


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    print(mse(y_left))
    print(mse(y_right))
    print(len(y_left))
    print(len(y_right))
    print(len(y_right))

    return (mse(y_left) * len(y_left) + mse(y_right) * len(y_right)) / (len(y_right) + len(y_left))
