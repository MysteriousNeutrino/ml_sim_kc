from typing import List
from typing import Tuple
from itertools import combinations
import copy


def extend_matches(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """

    :param pairs:
    :return:
    """
    elements = copy.deepcopy(pairs)
    print(elements)
    for pair in combinations(elements, 2):
        print("pairs: ", pair[0], pair[1])
        result = set(pair[0]) ^ set(pair[1])
        print(result)
        if len(result) == 2:
            elements.append(tuple(result))
    for pair in combinations(elements, 2):
        print("pairs: ", pair[0], pair[1])
        result = set(pair[0]) ^ set(pair[1])
        print(result)
        if len(result) == 2:
            elements.append(tuple(result))
    sorted_pairs = sorted([tuple(sorted(pair)) for pair in elements], key=lambda x: (x[0], x[1]))
    return sorted(list(set(sorted_pairs)))

# print((extend_matches([(1, 2), (2, 3), (5, 3), (4, 6), (6, 7), (8, 9)])))
