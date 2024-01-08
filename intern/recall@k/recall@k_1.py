from typing import List
import pandas as pd
import numpy as np
import sklearn


def sort_scores(scores: List[float]):
    """
    sort
    :param scores:
    :return:
    """
    return sorted(scores)


# def confusion_matrix(labels: List[int], scores: List[float], k) -> tuple:
#     df = pd.DataFrame({'True': labels, 'Predicted': sort_scores(scores)}).iloc[:k]
#
#     TN, FP, FN, TP = np.array(df.groupby(["True", "Predicted"])
#                               .value_counts().reset_index()["count"])
#
#     return TN, FP, FN, TP


def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """

    :param labels:
    :param scores:
    :param k:
    :return:
    """
    TP = 0
    FN = 0

    # Проходим по элементам исходных данных
    for true_label, score in zip(labels[:k], scores[:k]):
        # Если метка истинно положительная (True Positive)
        if true_label == 1:
            # Если оценка выше порогового значения 0.5
            if score >= 0.5:
                TP += 1
            # Если оценка ниже порогового значения
            else:
                FN += 1

    # Расчет полноты (recall)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    return recall


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """

    :param labels:
    :param scores:
    :param k:
    :return:
    """
    df = pd.DataFrame({'True': labels, 'Predicted': sorted(scores)}).iloc[:k]

    tp = df[(df['True'] == 1) & (df['Predicted'] == 1)].shape[0]
    fp = df[(df['True'] == 0) & (df['Predicted'] == 1)].shape[0]
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0
    return precision


def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """

    :param labels:
    :param scores:
    :param k:
    :return:
    """
    df = pd.DataFrame({'True': labels, 'Predicted': sorted(scores)}).iloc[:k]

    tn = df[(df['True'] == 0) & (df['Predicted'] == 0)].shape[0]
    fp = df[(df['True'] == 0) & (df['Predicted'] == 1)].shape[0]
    try:
        specificity = tn / (tn + fp)
    except ZeroDivisionError:
        return 0.0
    return specificity


def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """

    :param labels:
    :param scores:
    :param k:
    :return:
    """
    df = pd.DataFrame({'True': labels, 'Predicted': sorted(scores)}).iloc[:k]

    tp = df[(df['True'] == 1) & (df['Predicted'] == 1)].shape[0]
    fp = df[(df['True'] == 0) & (df['Predicted'] == 1)].shape[0]
    fn = df[(df['True'] == 1) & (df['Predicted'] == 0)].shape[0]
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return 0.0
    return f1
