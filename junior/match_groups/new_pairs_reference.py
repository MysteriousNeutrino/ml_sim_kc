from typing import List
from typing import Tuple


def extend_matches(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Add new pairs based on existing ones"""
    # Collect all unique ids
    unique_ids = set()
    for x, y in pairs:
        unique_ids.add(x)
        unique_ids.add(y)
    # Create a dictionary of existing pairs
    pairs_dict = {x: {x} for x in unique_ids}
    for x, y in pairs:
        united_pairs = pairs_dict[x] | pairs_dict[y]
        for z in united_pairs:
            pairs_dict[z].update(united_pairs)
    # Create a list of new pairs
    new_pairs = []
    for x, x_pairs in pairs_dict.items():
        for y in x_pairs:
            new_pairs.append((x, y))
    # Drop self-pairs and duplicates
    new_pairs = [pair for pair in new_pairs if pair[0] < pair[1]]

    # Sort the list of new pairs
    new_pairs = sorted(new_pairs)

    return new_pairs


print((extend_matches([(1, 2), (2, 3), (5, 3), (4, 6), (6, 7), (8, 9)])))