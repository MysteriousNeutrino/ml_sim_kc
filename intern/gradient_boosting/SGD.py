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
            verbose=False,
            subsample_size=0.5,
            replace=False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.base_pred_: float = None
        self.trees_: list = []
        self.loss_function: Callable = None
        self.subsample_size: float = subsample_size
        self.replace = replace

    def _mse(self, y_true, y_pred):
        loss = float(np.mean((y_pred - y_true) ** 2))
        grad = y_pred - y_true
        return loss, grad

    def _subsample(self, X, y=None):
        sample_indexes = np.random.choice(len(X), size=int(X.shape[0] * self.subsample_size), replace=self.replace)
        sub_X = X[sample_indexes]

        if y is not None:
             sub_y = y[sample_indexes]
        else:
            sub_y = None
        return sub_X, sub_y

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
        #     case "mse":
        #         self.loss_function = self._mse
        #     case function if callable(function):
        #         self.loss_function = function
        #     case _:
        #         raise AttributeError("unknown loos function")
        # 1
        self.base_pred_ = float(np.mean(y))

        if self.loss is None:
            self.loss_function = self._mse
        elif self.loss in ["mse"]:
            self.loss_function = self._mse
        elif callable(self.loss):
            self.loss_function = self.loss
        y_pred = self.base_pred_
        for i in range(self.n_estimators):
            # 2
            y_grad = self.loss_function(y, y_pred)[1]
            antigrad = -y_grad
            # 3
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split)
            # print(X.shape)
            # print("after tree: ", X.shape, y_grad.size)
            tree.fit(*(self._subsample(X, antigrad)))
            # 4
            y_pred = y_pred + tree.predict(X) * self.learning_rate
            # 5
            # print("y_pred in end of loop: ",len(y_pred))
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
        predictions = np.full(len(X), self.base_pred_)
        for tree in self.trees_:
            predictions = predictions + self.learning_rate * tree.predict(X)

        return predictions


#
#
# df = pd.read_csv(
#     r"C:\Users\Neesty\PycharmProjects\ml_sim_kc"
#     r"\intern\decision tree\load-delay_days_decision_tree_1000.csv")
# X = df.drop(columns=['delay_days']).values
# y = df['delay_days'].values
#
# model = GradientBoostingRegressor(max_depth=5, n_estimators=300, min_samples_split=5)
#
# model = model.fit(X[:1000], y[:1000])
# print(model.predict(X[500:510]))
# print(y[500:510])
