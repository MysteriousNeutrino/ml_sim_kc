import numpy as np


def smape(y_true: np.array, y_pred: np.array) -> float:
    with np.errstate(divide='ignore'):
        return np.mean(np.where(y_true + y_pred != 0, 2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)), 0))


