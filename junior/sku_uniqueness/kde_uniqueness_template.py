"""Solution's template for user."""
import numpy as np
from sklearn.neighbors import KernelDensity

def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    kde = KernelDensity(kernel='gaussian').fit(embeddings)

    kde_score = kde.score_samples(embeddings)
    uniqueness_score = 1 / np.exp(kde_score)
    return uniqueness_score
