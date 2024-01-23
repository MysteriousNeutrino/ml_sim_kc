import numpy as np
from typing import Tuple


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Tuple[int, int, int, int]:
    """Calculate confusion matrix."""

    y_pred_binary = (y_pred >= threshold).astype(int)

    TP = int(np.sum((y_true == 1) & (y_pred_binary == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred_binary == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred_binary == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred_binary == 0)))
    return TP, TN, FP, FN


def accuracy(TP: int, TN: int, FP: int, FN: int) -> float:
    """Calculate accuracy."""
    return (TP + TN) / (TP + TN + FP + FN)


def precision(TP: int, FP: int) -> float:
    """Calculate precision."""
    return TP / (TP + FP)


def recall(TP: int, FN: int) -> float:
    """Calculate recall."""
    return TP / (TP + FN)


def f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score."""
    return (2 * precision * recall) / (precision + recall)


def test():
    """Test function."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.6, 0.4, 0.5, 0.7])
    threshold = 0.5
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred, threshold)

    assert TP == 5
    assert TN == 4
    assert FP == 1
    assert FN == 0

    assert np.allclose(accuracy(TP, TN, FP, FN), 0.9)

    pr = precision(TP, FP)
    re = recall(TP, FN)
    assert np.allclose(pr, 0.8333333333333334)
    assert np.allclose(re, 1)
    assert np.allclose(f1_score(0.8333333333333334, 1), 0.9090909090909091)
    print("All tests passed.")


if __name__ == "__main__":
    test()
