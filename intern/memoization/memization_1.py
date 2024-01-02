from typing import Callable
import json
from functools import lru_cache


def memoize(func: Callable) -> Callable:
    """Memoize function"""
    cache = {}

    def wrapper(*args, **kwargs):
        # Convert dictionaries to hashable tuples
        args_hashable = tuple(args)
        kwargs_hashable = tuple(sorted(kwargs.items()))
        key = (args_hashable, kwargs_hashable)

        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    return wrapper
