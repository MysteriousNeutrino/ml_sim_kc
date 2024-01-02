"""
Функция average normalized_dcg в двух вариантах
"""
from typing import List

import numpy as np


def avg_ndcg(list_relevances: List[List[float]], k: int, method: str = 'standard') -> float:
    """Average nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values\
        `standard` - adds weight to the denominator\
        `industry` - adds weights to the numerator and denominator\
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    methodology = {"standard": lambda x, index: x / np.log2(index + 2),
                   "industry": lambda x, index: (2 ** x - 1) / np.log2(index + 2)}

    if method not in methodology:
        raise ValueError()
    score = []
    for relevance in list_relevances:
        dcg_score = []
        for index, x in enumerate(np.array(relevance[:k])):
            i_score = methodology[method](x, index)
            dcg_score.append(i_score)
        idcg_score = []
        for index, x in enumerate(np.array(sorted(relevance, reverse=True))[:k]):
            i_score = methodology[method](x, index)
            idcg_score.append(i_score)
        rel_score = sum(dcg_score) / sum(idcg_score)
        score.append(rel_score)

    return np.mean(score)
