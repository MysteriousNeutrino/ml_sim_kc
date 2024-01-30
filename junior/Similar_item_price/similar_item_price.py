from typing import Dict
from typing import Dict, Tuple, List
from itertools import combinations

import copy

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


class SimilarItems:
    def __init__(self, embeddings, prices):
        self.embeddings = embeddings
        self.prices = prices

    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Dict[Tuple[int, int], float]:
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """
        keys = embeddings.keys()
        combination = combinations(keys, 2)
        pair_sims = {}
        for left, right in combination:
            left_elements = np.array(embeddings.get(left)).reshape(1, -1)
            right_elements = np.array(embeddings.get(right)).reshape(1, -1)
            cos_sim = cosine_similarity(left_elements, right_elements)
            pair_sims[(left, right)] = round(cos_sim[0, 0], 8)
        return pair_sims

    @staticmethod
    def knn(
            sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        sorted_dict = dict(sorted(sim.items(), key=lambda item: item[1], reverse=True))
        result_dict = {}
        for key, value in sorted_dict.items():
            if result_dict.get(key[0]) is None:
                result_dict[key[0]] = []
            if result_dict.get(key[1]) is None:
                result_dict[key[1]] = []

            if len(result_dict.get(key[0])) < top:
                result_dict[key[0]].append((key[1], value))
            if len(result_dict.get(key[1])) < top:
                result_dict[key[1]].append((key[0], value))
        return result_dict

    @staticmethod
    def knn_price(
            knn_dict: Dict[int, List[Tuple[int, float]]],
            prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        knn_price_dict = copy.deepcopy(knn_dict)

        # for key, value_list in knn_price_dict.items():
        #     for i, (inner_key, inner_value) in enumerate(value_list):
        #         new_value = prices.get(inner_key, 0.0)
        #         knn_price_dict[key][i] = (inner_key, inner_value, new_value)

        for key, value_list in knn_price_dict.items():
            norm_weight = sum(map(lambda t: t[-1] + 1, value_list))

            for i, (inner_key, inner_value) in enumerate(value_list):
                price = prices.get(inner_key, 0.0)
                knn_price_dict[key][i] = ((inner_value + 1) / norm_weight) * price

            knn_price_dict[key] = round(sum(knn_price_dict.get(key)), 2)

        return knn_price_dict

    @staticmethod
    def transform(
            embeddings: Dict[int, np.ndarray],
            prices: Dict[int, float],
            top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        knn_price_dict = SimilarItems.knn_price(
            SimilarItems.knn(
                SimilarItems.similarity(embeddings), top), prices)
        return knn_price_dict

# embeddings = {
#     1: np.array([-26.57, -76.61, 81.61, -9.11, 74.8, 54.23, 32.56, -22.62, -72.44, -82.78]),
#     2: np.array([-55.98, 82.87, 86.07, 18.71, -18.66, -46.74, -68.18, 60.29, 98.92, -78.95]),
#     3: np.array([-27.97, 25.39, -96.85, 3.51, 95.57, -27.48, -80.27, 8.39, 89.96, -36.68]),
#     4: np.array([-37.0, -49.39, 43.3, 73.36, 29.98, -56.44, -15.91, -56.46, 24.54, 12.43]),
#     5: np.array([-22.71, 4.47, -65.42, 10.11, 98.34, 17.96, -10.77, 2.5, -26.55, 69.16])
# }
#
# prices = {
#     1: 100.5,
#     2: 12.2,
#     3: 60.0,
#     4: 11.1,
#     5: 245.2
# }
#
# print(SimilarItems.transform(embeddings,prices, 3))
