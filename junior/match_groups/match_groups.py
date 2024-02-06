from typing import List, Tuple, Union


def extend_matches(groups: List[Union[Tuple[int, ...], List[int]]]) -> List[Tuple[int, ...]]:
    """Add new groups based on existing ones"""
    # Collect all unique ids
    unique_ids = set()
    for group in groups:
        unique_ids.update(group)

    # Create a dictionary of existing groups
    groups_dict = {x: set([x]) for x in unique_ids}
    for group in groups:
        united_groups = set().union(*[groups_dict[x] for x in group])
        for z in united_groups:
            groups_dict[z].update(united_groups)

    # Create a list of new groups
    new_groups = []
    for x, x_groups in groups_dict.items():
        new_groups.append(tuple(sorted(x_groups)))

    # Drop self-groups and duplicates
    new_groups = sorted(list(set(new_groups)))
    return new_groups


# Пример использования с группами элементов разного размера
# print(extend_matches([(1, 2), (2, 3), (5, 3), (4, 6), (6, 7), (8, 9)]))
# print(extend_matches([(1, 2, 3), (2, 3, 4), (5, 3, 7), (4, 6, 8), (6, 7, 9), (8, 9, 10)]))
# print(extend_matches([(5, 3, 4, 8), (1, 2), (7, 2)]))
# print(extend_matches([(1, 2), (3, 4), (2, 5)]))
