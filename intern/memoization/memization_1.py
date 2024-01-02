import json
from typing import Callable, Any


def memoize(func: Callable) -> Callable:
    """Optimized memoize function"""
    cache = {}
    json_dumps = json.dumps

    def wrapper(*args, **kwargs) -> Any:
        key = json_dumps((args, kwargs), sort_keys=True)

        result = cache.get(key)
        if result is not None:
            return result

        result = func(*args, **kwargs)
        cache[key] = result

        return result

    return wrapper

