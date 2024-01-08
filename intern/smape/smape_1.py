import numpy as np


def smape(y_true: np.array, y_pred: np.array):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)

    # Заменяем деление на ноль на ноль
    result = np.where(denominator != 0, numerator / denominator, 0)

    return np.mean(result)
