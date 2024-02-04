"""Solution for Smartlink AB 2: Splitting"""
from typing import List, Tuple
import hashlib


class Experiment2:
    """Experiment class. Contains the logic for assigning users to groups."""

    def __init__(
        self,
        experiment_id: int,
        groups: Tuple[str] = ("A", "B"),
        group_weights: List[float] = None,
    ):
        self.experiment_id = experiment_id
        self.groups = groups
        self.group_weights = group_weights
        self.salt = hashlib.sha256(str(self.experiment_id).encode("utf-8")).hexdigest()

        if self.group_weights is not None:
            msg = (
                "The number of groups and the number of group weights must be the same."
            )
            assert len(self.group_weights) == len(self.groups), msg

            msg = "Group weights must be non-negative."
            assert all(weight >= 0 for weight in self.group_weights), msg

            msg = "The sum of group weights must be 1."
            assert sum(self.group_weights) == 1, msg

        if self.group_weights is None:
            self.group_weights = [1 / len(self.groups)] * len(self.groups)

    def group(self, click_id: int) -> Tuple[int, str]:
        """Assigns a click to a group.

        Parameters
        ----------
        click_id: int :
            id of the click

        Returns
        -------
        Tuple[int, str] :
            group id and group name
        """

        weights = [int(weight * 100) for weight in self.group_weights]

        hash_input = str(click_id) + self.salt
        hashed = int(hashlib.sha256(hash_input.encode("utf-8")).hexdigest(), 16)
        modulo = hashed % sum(weights)

        group = 0
        weight_cumsum = 0
        for i, weight in enumerate(weights):
            weight_cumsum += weight
            if modulo < weight_cumsum:
                group = i
                break

        return group, self.groups[group]

#
# ex1 = Experiment2(experiment_id=1, groups=("A", "B"), group_weights=None)
# ex2 = Experiment2(experiment_id=1, groups=("A", "B"), group_weights=None)
# ex3 = Experiment2(experiment_id=2, groups=("A", "B"), group_weights=[0.1, 0.9])
# ex4 = Experiment2(experiment_id=3, groups=("AAA", "BBB", "CCC"), group_weights=[0.1, 0.1, 0.8])
# # ex5 = Experiment(experiment_id=, groups=, group_weights=)
# # ex6 = Experiment(experiment_id=, groups=, group_weights=)
#
#
# print(ex1.group(1))
# print(ex2.group(1))
# print(ex3.group(1))
# print(ex4.group(1))
#
# group_numbers = []

