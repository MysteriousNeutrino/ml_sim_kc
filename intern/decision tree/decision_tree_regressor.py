from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    # YOUR CODE HERE: add the required attributes
    threshold: float = None
    n_samples: int = None
    value: int = None
    mse: float = None
    left: Node = None
    right: Node = None


@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        if len(y) == 0:
            return 0.0
        return float(np.mean((y - np.mean(y)) ** 2))

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weighted mse criterion for a two given sets of target values"""
        # if len(y_left) == 0 or len(y_right) == 0:
        #     return 0.0
        num = self._mse(y_left) * y_left.size + self._mse(y_right) * y_right.size
        den = y_left.size + y_right.size
        return num / den

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        node_size = y.size
        n_features_ = X.shape[1]
        if node_size < 2:
            return None, None

        node_mse = self._mse(y)
        best_mse = node_mse
        best_idx, best_thr = None, None

        for idx in range(n_features_):
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                left = y[X[:, idx] <= thr]
                right = y[X[:, idx] > thr]

                if left.size == 0 or right.size == 0:
                    continue

                weihted_mse = self._weighted_mse(left, right)
                if weihted_mse < best_mse:
                    best_mse = weihted_mse
                    best_idx = idx
                    best_thr = thr

        return best_idx, best_thr

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        # print("_split_node y: ", y)
        node = Node()
        print(X)
        print(y)
        if depth == self.max_depth or len(np.unique(y)) == 1:
            node.n_samples = y.size
            node.value = int(round(np.mean(y)))
            node.mse = self._mse(y)
            return node

        best_idx, best_thr = self._best_split(X, y)
        node.feature = best_idx
        node.threshold = best_thr
        if best_idx is not None and best_thr is not None:
            left_mask = X[:, best_idx] <= best_thr
            right_mask = X[:, best_idx] > best_thr


            node.n_samples = len(y)
            node.value = int(round(np.mean(y)))

            # Добавлены проверки на пустые подвыборки
            node.mse = self._mse(y)
            print(node)
            node.left = self._split_node(X[left_mask], y[left_mask], depth + 1)
            node.right = self._split_node(X[right_mask], y[right_mask], depth + 1)
        else:
            return node
        return node



# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24],
#               [25, 26, 27], [28, 29, 30]])
# y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# df = pd.read_csv(r"C:\Users\Neesty\PycharmProjects\ml_sim_kc\intern\decision tree\load-delay_days_decision_tree_1000.csv")
# df = df[:50]
# X = df.drop(columns=['delay_days']).values
# y = df['delay_days'].values
#
# decisionTreeRegressor = DecisionTreeRegressor(max_depth=2)
#
# model = decisionTreeRegressor.fit(X, y)
#
# print(model.tree_)