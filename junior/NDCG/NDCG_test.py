"""
Функция DCG в двух вариантах
"""
from typing import List

import numpy as np


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """

    dcg_score = []
    idcg_score = []
    if method == "standard":
        for index, x in enumerate(np.array(relevance[:k])):
            i_score = x / np.log2(index + 2)
            print(i_score)
            dcg_score.append(i_score)
        for index, x in enumerate(sorted(np.array(relevance[:k]), reverse=True)):
            i_score = x / np.log2(index + 2)
            print(i_score)
            idcg_score.append(i_score)
    elif method == "industry":
        for index, x in enumerate(np.array(relevance[:k])):
            i_score = (2 ** x - 1) / np.log2(index + 2)
            print(i_score)
            dcg_score.append(i_score)
        for index, x in enumerate(np.array(sorted(relevance, reverse=True))[:k]):
            i_score = (2 ** x - 1) / np.log2(index + 2)
            print(i_score)
            idcg_score.append(i_score)
    score = np.array(dcg_score).sum() / np.array(idcg_score).sum()
    return score


print(discounted_cumulative_gain([0.99, 0.94, 0.74, 0.88, 0.71, 0.68], 5, 'standard'))
print("---------------")
print(discounted_cumulative_gain([0.99, 0.94, 0.74, 0.88, 0.71, 0.68], 5, 'industry'))
