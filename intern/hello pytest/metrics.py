from typing import List


def profit(revenue: List[float], costs: List[float]) -> float:
    return sum(revenue) - sum(costs)


def margin(revenue: List[float], costs: List[float]) -> float:
    return (sum(revenue) - sum(costs)) / sum(revenue)


def markup(revenue: List[float], costs: List[float]) -> float:
    return (sum(revenue) - sum(costs)) / sum(costs)


print(margin([1, 2, 3], [1, 1, 1]))
print(markup([1, 2, 3], [1, 1, 1]))