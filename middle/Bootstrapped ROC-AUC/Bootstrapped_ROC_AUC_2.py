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
    n = len(X)
    aucs = []
    y_pred_proba = classifier.predict_proba(X)
    for _ in range(n_bootstraps):

        indices = np.random.choice(n, n, replace=True)
        boot_y_pred = y_pred_proba[indices][:, 1]
        boot_y_true = y[indices]
        if len(np.unique(boot_y_true)) == 1:
            continue
        # Compute ROC-AUC
        auc = roc_auc_score(boot_y_true, boot_y_pred)
        aucs.append(auc)

    lcb, ucb = np.quantile(aucs, q=[(1 - conf) / 2, 1 - (1 - conf) / 2])

    return (lcb, ucb)
