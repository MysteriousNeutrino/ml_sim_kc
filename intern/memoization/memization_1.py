from typing import Callable


def memoize(func: Callable) -> Callable:
    """Memoize function"""
    cache = {}

    def decorate(*args, **kwargs):
        # Преобразование kwargs в кортеж кортежей для хеширования
        sorted_kwargs = tuple(sorted(kwargs.items()))
        key = (tuple(args), hash(sorted_kwargs))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return decorate
