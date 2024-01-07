def word_count(texts):
    count = {}
    for text in texts:
        for word in text.split():
            count[word] = count.get(word, 0) + 1
    return count


print(word_count("hello world!"))
print(word_count("hello world!"))


def word_count_2(batch, count=None):
    if count is None:
        count = {}
    for text in batch:
        for word in text.split():
            count[word] = count.get(word, 0) + 1
    return count


print(word_count_2("hello world!"))
print(word_count_2("hello world!"))
