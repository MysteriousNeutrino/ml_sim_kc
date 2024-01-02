"""Returns confidence bounds of the ROC-AUC"""
from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score


def roc_auc_ci(
        classifier: ClassifierMixin,
        X: np.ndarray,
        y: np.ndarray,
        conf: float = 0.95,
        n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""

    aucs = []

    for _ in range(n_bootstraps):
        sample_X = np.random.choice(X.flatten(), X.flatten().size).reshape(X.shape)
        sample_y = np.random.choice(y, len(y))

        classifier.predict(sample_X)


        # Predict probabilities

        y_pred_proba = classifier.predict_proba(X)[:,1]

        # Compute ROC-AUC
        auc = roc_auc_score(y, y_pred_proba)
        aucs.append(auc)

    alpha = 1.0 - conf
    lcb, ucb = np.quantile(aucs, q=[alpha / 2, 1 - alpha / 2])

    return (lcb, ucb)
