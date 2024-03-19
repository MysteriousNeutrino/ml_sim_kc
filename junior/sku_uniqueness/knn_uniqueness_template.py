"""Solution's template for user."""
import numpy as np
from  sklearn.metrics.pairwise import euclidean_distances


def knn_uniqueness(embeddings: np.ndarray, num_neighbors: int) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on knn.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 
    num_neighbors: int :
        number of neighbors to estimate uniqueness    

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    distance = euclidean_distances(embeddings[:, :num_neighbors])
    uniqueness = np.mean(distance, axis=1)
    return uniqueness


