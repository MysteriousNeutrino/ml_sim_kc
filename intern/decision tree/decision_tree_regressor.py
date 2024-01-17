from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    # YOUR CODE HERE: add the required attributes
    threshold: float = None
    n_samples: int = None
    value: int = None
    mse: int = None
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
        return np.mean((y - np.mean(y)) ** 2)

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weighted mse criterion for a two given sets of target values"""
        num = self._mse(y_left) * y_left.size + self._mse(y_right) * y_right.size
        den = y_left.size + y_right.size
        return num / den

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        best_split = {}
        for feature in range(X.shape[1]):
            split_elements = X[:, feature]

            best_split_1_future = {}
            for i in split_elements:
                indexes_below_split_value = np.where(split_elements <= i)[0]
                indexes_above_split_value = np.where(split_elements > i)[0]

                y_left = np.array(y[indexes_below_split_value])
                y_right = np.array(y[indexes_above_split_value])

                mse = self._weighted_mse(y_left, y_right)
                best_split_1_future[i] = mse

            best_split[feature] = min(best_split_1_future, key=best_split_1_future.get)

        best_feature, best_thr = min(best_split.items(), key=lambda x: x[1])

        return best_feature, best_thr

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        node = Node()
        node.n_samples = y.size
        node.value = np.mean(y)
        node.mse = self._mse(y)
        if depth == self.max_depth or y.size >= self.min_samples_split:
            return node
        node.feature, node.threshold = self._best_split(X, y)
        depth += 1
        if node.threshold is not None:
            node.left = X[X[:, node.feature] <= node.threshold], y[X[:, node.feature] <= node.threshold]
            self._split_node(node.left[0], node.left[1], depth)
            node.right = X[X[:, node.feature] > node.threshold], y[X[:, node.feature] > node.threshold]
            self._split_node(node.right[0], node.right[1], depth)
        else:
            return node
        return node
