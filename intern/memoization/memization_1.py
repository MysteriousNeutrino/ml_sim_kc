from typing import Any
from typing import Callable


def memoize(func: Callable) -> Callable:
    """Memoize function"""
    cache = {}

    def memoized(*args, **kwargs) -> Any:
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized
