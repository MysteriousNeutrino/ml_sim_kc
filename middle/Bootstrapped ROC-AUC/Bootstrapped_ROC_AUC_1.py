from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score
from scipy.stats import bootstrap


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
        indices = np.random.choice(len(X))
        sample_X = X[indices]
        sample_y = y[indices]

        classifier.fit(sample_X, sample_y)

        # Predict probabilities
        y_pred_proba = classifier.predict_proba(X)[:, 1]

        # Compute ROC-AUC
        auc = roc_auc_score(y, y_pred_proba)
        aucs.append(auc)

    alpha = 1.0 - conf
    p = ((1.0 - alpha) / 2.0) * 100

    lcb = max(0.0, np.percentile(aucs, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    ucb = min(1.0, np.percentile(aucs, p))

    return (lcb, ucb)
