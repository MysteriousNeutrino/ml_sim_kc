from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


class BaseSelector(ABC):
    """
    abstract BaseSelector
    """

    @abstractmethod
    def fit(self, X, y):
        """
        fit method
        :param X:
        :param y:
        :return:
        """
        pass

    def transform(self, X):
        """
        transform
        :param X:
        :return:
        """
        return X[self.high_corr_features]

    def fit_transform(self, X, y):
        """
        fit_transform
        :param X:
        :param y:
        :return:
        """
        self.fit(X, y)
        return self.transform(X)

    @property
    def n_features_(self):
        """
        n_features_
        :return:
        """

        return len(self.high_corr_features)

    @property
    def original_features_(self):
        """
        original_features_
        :return:
        """
        return self.original_features

    @property
    def selected_features_(self):
        """
        selected_features_
        :return:
        """
        return self.high_corr_features


@dataclass
class PearsonSelector(BaseSelector):
    """
    PearsonSelector
    """
    threshold: float = 0.5

    def fit(self, X, y) -> PearsonSelector:
        """
        fit
        :param X:
        :param y:
        :return:
        """
        # Correlation between features and target
        super().fit(X, y)
        corr = pd.concat([X, y], axis=1).corr(method="pearson")
        corr_target = corr.iloc[:-1, -1]

        self.original_features = X.columns.tolist()
        self.high_corr_features = corr_target[
            abs(corr_target) >= self.threshold
            ].index.tolist()

        return self


@dataclass
class SpearmanSelector(BaseSelector):
    """
    SpearmanSelector
    """
    threshold: float = 0.5

    def fit(self, X, y) -> SpearmanSelector:
        """
        fit
        :param X:
        :param y:
        :return:
        """
        super().fit(X, y)
        corr = pd.concat([X, y], axis=1).corr(method="spearman")
        corr_target = corr.iloc[:-1, -1]
        self.original_features = X.columns.tolist()
        self.high_corr_features = corr_target[
            abs(corr_target) >= self.threshold
            ].index.tolist()

        return self


@dataclass
class VarianceSelector(BaseSelector):
    """
    VarianceSelector
    """
    min_var: float = 0.4

    def fit(self, X, y=None) -> VarianceSelector:
        """
        fit
        :param X:
        :param y:
        :return:
        """
        super().fit(X, y)
        variances = np.var(X, axis=0)
        self.original_features = X.columns.tolist()
        self.high_corr_features = X.columns[variances > self.min_var].tolist()
        return self
