import utils


def test_word_count():
    """
    Check stable work of the function
    :return:
    """

    assert utils.word_count("hello world!") == \
           {'h': 1, 'e': 1, 'l': 3, 'o': 2, 'w': 1, 'r': 1, 'd': 1, '!': 1}


def test_word_count_tricky():
    """
    Check that the function is not mutating the default argument
    :return:
    """
    assert utils.word_count("hello world!") == \
           {'h': 1, 'e': 1, 'l': 3, 'o': 2, 'w': 1, 'r': 1, 'd': 1, '!': 1}
    assert utils.word_count("hello world!") == \
           {'h': 1, 'e': 1, 'l': 3, 'o': 2, 'w': 1, 'r': 1, 'd': 1, '!': 1}
