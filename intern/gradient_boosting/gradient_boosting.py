from typing import Callable
from scipy.optimize import approx_fprime
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(
            self,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            loss=None,
            verbose=False, ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose

    base_pred_: float = None
    trees_: list = []
    loss_function: Callable = None

    def _mse(self, y_true, y_pred):
        loss = float(np.mean((y_pred - y_true) ** 2))
        grad = y_pred - y_true
        return loss, grad

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """

        # match self.loss:
        #     case None:
        #         self.loss_function = self._mse
        #     case "mse" | "mae":  # "mse" | "mae" для нескольких вариантов
        #         self.loss_function = self._mse
        #     case function if callable(function):
        #         self.loss_function = function
        #     case _:
        #         raise AttributeError("unknown loos function")

        if self.loss is None:
            self.loss_function = self._mse
        elif self.loss in ["mse"]:
            self.loss_function = self._mse
        elif callable(self.loss):
            self.loss_function = self.loss

        # 1
        self.base_pred_ = float(np.mean(y))
        y_pred = self.base_pred_  # np.full((y.size, 1), self.base_pred_)
        for i in range(self.n_estimators):
            # 2
            y_grad = self.loss_function(y, y_pred)[1]  # approx_fprime(y_pred, lambda pred: self.loss_function(y, pred), epsilon=1e-6)
            # print(y_grad)
            # 3
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split)

            tree.fit(X, -y_grad)
            # 4
            y_pred = y_pred + tree.predict(X) * self.learning_rate
            # 5
            self.trees_.append(tree)
        return self

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.

        """
        predictions = self.base_pred_
        for tree in self.trees_:
            predictions = predictions + self.learning_rate * tree.predict(X)

        return predictions


df = pd.read_csv(
    r"C:\Users\Neesty\PycharmProjects\ml_sim_kc"
    r"\intern\decision tree\load-delay_days_decision_tree_1000.csv")
df = df[:10]
X = df.drop(columns=['delay_days']).values
y = df['delay_days'].values


model = GradientBoostingRegressor(max_depth=2)

model.fit(X, y)

model.predict(X)
