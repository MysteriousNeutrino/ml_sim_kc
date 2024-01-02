"""
    Целевое состояние -- небольшая переоценка
    Асинхроная метрика.
"""
import numpy as np


def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    """

    Больше штрафуем за недооценку и за большие отклонения.

    - y_true: массив истинных значений
    - y_pred: массив предсказанных значений
    :return: metric
    """

    error = np.log(np.cosh(y_true - y_pred)) * (((y_true - y_pred) / y_pred) ** 2)

    return error.mean()
