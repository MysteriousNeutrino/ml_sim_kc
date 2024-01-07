import metrics


def test_profit() -> None:
    """
tests for profit function
    :return:
    """
    assert metrics.profit([1, 2, 3], [1, 1, 1]) == 3


def test_margin() -> None:
    """
    tests for margin function
    :return:
    """
    assert metrics.margin([1, 2, 3], [1, 1, 1]) == 0.5


def test_markup() -> None:
    """
    tests for markup function
    :return:
    """
    assert metrics.markup([1, 2, 3], [1, 1, 1]) == 1
