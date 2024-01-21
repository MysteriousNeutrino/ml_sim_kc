import random


def username_generator(n, first_names=None, last_names=None):
    """
    username_generator
    :param n:
    :param first_names:
    :param last_names:
    :return:
    """
    if first_names is None:
        first_names = ["John", "Jane", "Alice"]
    if last_names is None:
        last_names = ["Doe", "Smith", "Johnson"]

    for user in range(n):
        yield {'id': user + 1,
               'first_name': random.choice(first_names),
               'last_name': random.choice(last_names)}


# Example of use
custom_first_names = ["Max", "Sophia", "Liam"]
custom_last_names = ["Miller", "Davis", "Garcia"]
for user in username_generator(3, custom_first_names, custom_last_names):
    print(user['id'], user['first_name'], user['last_name'])


# Example of output:
# 1 Max Garcia
# 2 Liam Davis
# 3 Max Miller


def data_generator(n):
    """
    data_generator
    :param n:
    :return:
    """
    for i in range(n):
        yield (i, random.randint(0, 100))


# Example of use
for data in data_generator(3):
    print(data)

# Example of output:
# (0, 49)
# (1, 27)
# (2, 88)
