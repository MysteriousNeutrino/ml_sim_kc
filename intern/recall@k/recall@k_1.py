"""
общий алгоритм
1. Отсортировать оценки(score) в порядке убывания
1.2 Отобрать по их индексам labels (k штук) -- top_k_labels
    - будут выступать в качестве наших предсказаний
1.3 Отобрать labels которые не попали в п1.2 -- non_top_k_labels
    - будут выступать в качестве того что не попало в наши предсказания
2. Взять метки(labels) для элементов из п.1.2 и п.1.3
3. Рассчитать нужные метрики из confusion matrix
    TP -- количеcтво 1 в top_k_labels
    TN -- количеcтво 0 в non_top_k_labels
    FP -- количеcтво 0 в top_k_labels
    FN -- количеcтво 1 в non_top_k_labels
    P  -- всего 1 в исходном массиве (labels)
    N  -- всего 0 в исходном массиве (labels)
4. Рассчитать нужные метрики performance metrics
"""
from typing import List
import pandas as pd
import numpy as np
import sklearn


def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """

    :param labels:
    :param scores:
    :param k:
    :return:
    """
    # Сортируем оценки в порядке убывания и выбираем топ-k
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    # Выбираем метки для топ-k оценок
    top_k_labels = [labels[i] for i in top_k_indices]

    # Рассчитываем recall@k
    true_positives = sum(top_k_labels)
    total_positives = sum(labels)

    recall = true_positives / total_positives if total_positives > 0 else 0

    return recall


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """

    :param labels:
    :param scores:
    :param k:
    :return:
    """
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    # Выбираем метки для топ-k оценок
    top_k_labels = [labels[i] for i in top_k_indices]

    # Рассчитываем precision@k
    true_positives = sum(top_k_labels)
    total_positives = sum(labels)
    precision = true_positives / k if k > 0 else 0
    return precision


def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """

    :param labels:
    :param scores:
    :param k:
    :return:
    """
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    # Выбираем метки для топ-k оценок
    top_k_labels = [labels[i] for i in top_k_indices]
    non_top_k_labels = [labels[i] for i in range(len(labels)) if i not in top_k_indices]
    # кол-во нулей не вошедших k самых вероятных предсказаний для 1
    true_negatives = len(non_top_k_labels) - sum(non_top_k_labels)
    # кол-во нулей среди предсказаний (для 1)
    false_positive = len(top_k_labels) - np.sum(top_k_labels)
    specificity = true_negatives / (false_positive + true_negatives) if (false_positive + true_negatives) > 0 else 0
    return specificity


def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """

    :param labels:
    :param scores:
    :param k:
    :return:
    """
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    # Выбираем метки для топ-k оценок
    top_k_labels = [labels[i] for i in top_k_indices]

    # Рассчитываем precision@k
    true_positives = sum(top_k_labels)
    total_positives = sum(labels)
    recall = true_positives / total_positives if total_positives > 0 else 0
    precision = true_positives / k if k > 0 else 0
    f1 = 2 * (recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
    return f1
