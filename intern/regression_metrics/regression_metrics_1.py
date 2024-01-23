import numpy as np


def mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """

    :param actual:
    :param predicted:
    :return:
    """
    return float(np.mean((actual - predicted) ** 2))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """

    :param actual:
    :param predicted:
    :return:
    """
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """

    :param actual:
    :param predicted:
    :return:
    """
    return float(np.mean(np.abs(actual - predicted)))


def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """

    :param actual:
    :param predicted:
    :return:
    """
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """

    :param actual:
    :param predicted:
    :return:
    """
    return 1 - (float((np.mean((actual - predicted) ** 2)))
                / float((np.mean((actual - np.mean(actual)) ** 2))))

def test():
    actual = np.array([3, -0.5, 2, 7])
    predicted = np.array([2.5, 0.0, 2, 8])

    assert np.allclose(mean_squared_error(actual, predicted), 0.375)
    assert np.allclose(root_mean_squared_error(actual, predicted), 0.6123724356957945)
    assert np.allclose(mean_absolute_error(actual, predicted), 0.5)
    assert np.allclose(
        mean_absolute_percentage_error(actual, predicted), 32.73809523809524
    )
    assert np.allclose(r_squared(actual, predicted), 0.9486081370449679)

    print("All tests passed.")


if __name__ == "__main__":
    pass
