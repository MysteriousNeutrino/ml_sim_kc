"""
tree visualization
"""
from __future__ import annotations

import json
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
        node = Node()
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
            # print("X: ", X)
            # print("left x: ",X[left_mask])
            # print("right x: ",X[right_mask])
            # print("-------------------------------------")
            node.n_samples = len(y)
            node.value = int(round(np.mean(y)))

            # Добавлены проверки на пустые подвыборки
            node.mse = self._mse(y)
            node.left = self._split_node(X[left_mask], y[left_mask], depth + 1)
            node.right = self._split_node(X[right_mask], y[right_mask], depth + 1)
        else:
            return node
        return node

    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        self.json = self._as_json(self.tree_)
        return json.dumps(self.json)

    def _as_json(self, node: Node) -> str:
        """Return the decision tree as a JSON string. Execute recursively."""
        if node.left is None and node.right is None:
            return {
                "value": int(node.value),
                "n_samples": int(node.n_samples),
                "mse": round(float(node.mse), 2)
            }
        return ({
            "feature": int(node.feature),
            "threshold": int(node.threshold),
            "n_samples": int(node.n_samples),
            "mse": round(float(node.mse), 2),
            "left": self._as_json(node.left),
            "right": self._as_json(node.right)
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        result = []
        for predict in X:
            result.append(self._predict_one_sample(predict))

        return np.array(result)

    def _predict_one_sample(self, features: np.ndarray) -> int:
        """Predict the target value of a single sample."""
        request_leafs = []
        for i in range(3):
            path = "self.tree_." + "".join(request_leafs)
            # print("path: ",path)
            # print("threshold: ", eval(path + 'threshold'))
            # print("feature: ", eval(path + 'feature'))
            # print(eval(path + 'feature'))
            if eval(path + 'threshold') is not None and eval(path + 'feature') is not None:
                if features[eval(path + 'feature')] > eval(path + 'threshold'):
                    request_leafs.append('right.')
                else:
                    request_leafs.append('left.')
            # print("self.tree_." + "".join(request_leafs) + "value")


        result = "self.tree_." + "".join(request_leafs) + "value"
        # print("result: ", eval(result))
        return eval(result)

# df = pd.read_csv(
#     r"C:\Users\Neesty\PycharmProjects\ml_sim_kc"
#     r"\intern\decision tree\load-delay_days_decision_tree_1000.csv")
# df = df[:10]
# X = df.drop(columns=['delay_days']).values
# y = df['delay_days'].values
#
# decisionTreeRegressor = DecisionTreeRegressor(max_depth=10)
#
# model = decisionTreeRegressor.fit(X, y)
# # print(X)
# print(f'{model.tree_}')
# print(model.predict(np.array([[9, 61, 41396, 5, 0, 0, 636, 6, 0, 483916, 120],[7, 55, 44396, 8, 0, 0, 200, 6, 0, 483916, 120]])))
# # print((model.as_json()))