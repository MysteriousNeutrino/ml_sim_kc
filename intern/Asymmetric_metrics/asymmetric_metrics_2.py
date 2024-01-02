"""
    Целевое состояние -- небольшая недоценка
    Асинхроная метрика.
"""
import numpy as np


def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    """

    Больше штрафуем за переоценку и за большие отклонения.

    - y_true: массив истинных значений
    - y_pred: массив предсказанных значений
    :return: metric
    """

    error = np.where(y_pred > y_true, 4 * (((y_true - y_pred) / y_pred) ** 2),
                     (((y_true - y_pred) / y_true) ** 2))
    return error.mean()
