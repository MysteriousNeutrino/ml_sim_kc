"""
https://lab.karpov.courses/learning/77/module/949/lesson/12166/35134/175231/
"""

import numpy as np


def turnover_error(y_true: np.array, y_pred: np.array, is_weighted: bool) -> float:
    """
    Целевое состояние -- небольшая переоценка
    Асинхроная метрика.
    Больше штрафуем за недооценку и за большие отклонения.

    :param y_true:
    :param y_pred:
    :param beta:
    :return: metric

    """
    weight = np.where(y_pred > y_true, 4 * (((y_true - y_pred) / y_pred) ** 2),
                      (((y_true - y_pred) / y_true) ** 2))
    #
    # def format_number(num):
    #     return "{:.10f}".format(num)
    #
    # weight = np.vectorize(format_number)(weight)
    if is_weighted:
        error = np.log(np.cosh(y_true - y_pred)) * weight
    else:
        error = np.log(np.cosh(y_true - y_pred))
    weighted_log_cosh = np.mean(error)
    print("---------")
    print("weight = ", weight)
    print("error = ", error)
    print("weighted_log_cosh = ", weighted_log_cosh)
    return error


# turnover_error(np.array([100, 100, 100, 100]), np.array([10, 150, 30, 500]), 2, True)
# turnover_error(np.array([100] * 4), np.array(np.linspace(10, 500, 4)), 1, True)
# print(np.array(np.linspace(10, 500, 5)))

# print(turnover_error(100, 100, True))

print(turnover_error(100, 90, True))
print(turnover_error(100, 110, True))

print(turnover_error(100, 50, True))
print(turnover_error(100, 150, True))
