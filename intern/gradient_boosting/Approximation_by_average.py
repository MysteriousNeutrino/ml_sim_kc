import numpy as np
import pandas as pd


class GradientBoostingRegressor:
    """Gradient boosting regressor."""
    base_pred_: int = None
    def fit(self, X, y):
        """Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        self.base_pred_ = np.mean(y)
        return self

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.

        """

        return np.full(X.shape[0], self.base_pred_)
