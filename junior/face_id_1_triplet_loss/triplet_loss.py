import numpy as np


def triplet_loss(
    anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 5.0
) -> float:
    """
    Computes the triplet loss using numpy.
    Using Euclidean distance as metric function.

    Args:
        anchor (np.ndarray): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (np.ndarray): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (np.ndarray): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        float: The triplet loss
    """
    point_distance_positive = np.linalg.norm(anchor - positive, axis=1)
    point_distance_negative = np.linalg.norm(anchor - negative, axis=1)
    L = np.maximum(0, point_distance_positive - point_distance_negative + margin)

    return np.mean(L)
