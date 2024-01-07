import metrics


def test_non_int_clicks():
    """
    check for clicks not int
    :return:
    """
    try:
        metrics.ctr(1.5, 2)
    except TypeError:
        pass
    else:
        raise AssertionError("Non int clicks not handled")


def test_non_int_views():
    """
    check for views not int
    :return:
    """
    try:
        metrics.ctr(2, 1.5)
    except TypeError:
        pass
    else:
        raise AssertionError("Non int views not handled")


def test_non_positive_clicks():
    """
    check for clicks not positive
    :return:
    """
    try:
        metrics.ctr(-2, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Non positive clicks not handled")


def test_non_positive_views():
    """
    check for views not positive
    :return:
    """
    try:
        metrics.ctr(2, -2)
    except ValueError:
        pass
    else:
        raise AssertionError("Non positive views not handled")


def test_clicks_greater_than_views():
    """
    check for views not less than clicks
    :return:
    """
    try:
        metrics.ctr(2, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Non views greater than clicks not handled")


def test_zero_views():
    """
    check for views is non-zero
    :return:
    """
    try:
        metrics.ctr(2, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Non non-zero views not handled")
