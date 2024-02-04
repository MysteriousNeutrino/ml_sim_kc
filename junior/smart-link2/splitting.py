from typing import Tuple, List
import random
import hashlib

import numpy as np


class Experiment:
    """Experiment class. Contains the logic for assigning users to groups."""

    def __init__(
            self,
            experiment_id: int,
            groups=("A", "B"),
            group_weights: List[float] = None,
    ):
        self.experiment_id = experiment_id
        self.groups = groups
        self.group_weights = group_weights

        # Define the salt for experiment_id.
        # The salt should be deterministic and unique for each experiment_id.
        # letters = string.ascii_letters
        # self.salt = ''.join(random.choice(letters) for _ in range(15))
        hash_object = hashlib.sha256(str(experiment_id).encode())
        # Получаем шестнадцатеричное значение хэша и используем его как строку
        self.salt = hash_object.hexdigest()

        # Define the group weights if they are not provided equaly distributed
        # Check input group weights. They must be non-negative and sum to 1.
        if group_weights is not None:
            if sum(group_weights) != 1:
                raise ValueError("the sum of group_weights not equal to 1")

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

        # Assign the click to a group randomly based on the group weights
        # Return the group id and group name
        # click_hash = hash(str(click_id) + str(self.salt))

        weights = [int(weight * 100) for weight in self.group_weights]
        hash_input = str(click_id) + self.salt
        hashed = int(hashlib.sha256(hash_input.encode("utf-8")).hexdigest(), 16)
        modulo = hashed % sum(weights)

        probability_range = [0] + np.cumsum(self.group_weights) * 100
        group_id = np.digitize(modulo, probability_range, right=False)

        return group_id, self.groups[group_id]

# ex1 = Experiment(experiment_id=1, groups=("A", "B"), group_weights=None)
# ex2 = Experiment(experiment_id=1, groups=("A", "B"), group_weights=None)
# ex3 = Experiment(experiment_id=2, groups=("A", "B"), group_weights=[0.1, 0.9])
# ex4 = Experiment(experiment_id=3, groups=("AAA", "BBB", "CCC"), group_weights=[0.1, 0.1, 0.8])
# # ex5 = Experiment(experiment_id=, groups=, group_weights=)
# # ex6 = Experiment(experiment_id=, groups=, group_weights=)
#
#
# print(ex4.group(1))
#
# group_numbers = []
#
# for i in range(1000):
#     group_numbers.append(ex4.group(i)[0])
#
# sns.histplot(group_numbers, bins=max(group_numbers)-min(group_numbers)+1, kde=False, color='blue', alpha=0.7)
# plt.title('Распределение номеров групп')
# plt.xlabel('Номер группы')
# plt.ylabel('Количество')
#
# # Отображаем график
# plt.show()
