import numpy as np


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Residuals"""
    return y_true - y_pred


def squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Squared errors"""
    return (y_true - y_pred) ** 2


def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """LogLoss terms"""
    if np.all(y_pred > 0) and np.all(y_pred < 1):
        return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    else:
        raise ValueError


def ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAPE terms"""
    if np.all(y_true >= 0) and np.all(y_pred >= 0):
        return 1 - np.where(y_true == 0, 0, y_pred / y_true)
    else:
        raise ValueError


def quantile_loss(
        y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.01
) -> np.ndarray:
    """Quantile loss terms"""
    return np.where(y_true >= y_pred, q * (y_true - y_pred), (1 - q) * (y_pred - y_true))


# # # print(1 - np.array([1]))
# # # print(np.log(0))
print(logloss(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])))
# #
# print(ape(np.array([-1, 2, 3]), np.array([2, 2, 4])))
