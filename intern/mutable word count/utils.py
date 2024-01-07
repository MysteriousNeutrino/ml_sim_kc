def word_count(batch, count=None):
    """
    a function that count the number of words in a text
    :param batch: text where you need to count the number of words
    :param count: batch size
    :return: frequency dictionary
    """
    count = count or {}  # alernative to: if count is None: count = {}

    for text in batch:
        for word in text.split():
            count[word] = count.get(word, 0) + 1
    return count
