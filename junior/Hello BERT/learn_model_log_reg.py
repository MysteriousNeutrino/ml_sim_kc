from typing import List

from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import cross_val_score, KFold


def evaluate(model, embeddings, labels, cv=5) -> List[float]:
    """

    :param model:
    :param embeddings:
    :param labels:
    :param cv:
    :return:
    """
    loss_scorer = make_scorer(log_loss, greater_is_better=True, needs_proba=True)

    # Выполняем кросс-валидацию с использованием функции потерь
    scores = cross_val_score(model, embeddings, labels, cv=KFold(n_splits=cv), scoring=loss_scorer)
    return scores
