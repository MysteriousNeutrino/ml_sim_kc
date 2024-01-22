from functools import reduce
from typing import List


def sales_with_tax(sales: List[float], tax_rate: float, threshold: float = 300) -> List[float]:
    """

    :param sales:
    :param tax_rate:
    :param threshold:
    :return:
    """
    return list(map(lambda x: x * (1 + tax_rate), (filter(lambda x: x > threshold, sales))))


print(sales_with_tax([100, 200, 300, 400, 500], 0.2))


def sum_sales(sales: List[float], threshold: float = 300) -> float:
    """

    :param sales:
    :param threshold:
    :return:
    """
    return float(reduce(lambda x, y: x + y, (filter(lambda x: x > threshold, sales))))


print(sum_sales([100, 200, 300, 400, 500], 300))


def average_age(ages: List[int], threshold: int = 30) -> float:
    """

    :param ages:
    :param threshold:
    :return:
    """
    return reduce(lambda x, y: x + y, (filter(lambda x: x > threshold, ages))) / len(
        list((filter(lambda x: x > threshold, ages))))


print(average_age([100, 200, 300, 400, 500], 300))


def increased_prices(prices: List[float],
                     increase_rate: float = 0.2,
                     threshold: int = 300) -> List[float]:
    """

    :param prices:
    :param increase_rate:
    :param threshold:
    :return:
    """
    return list(filter(lambda x: x >= threshold, map(lambda x: x * (1 + increase_rate), prices)))


print(increased_prices([100, 200, 300, 400, 500], 0.2, threshold=300))


def weighted_sale_price(sales: List[float]) -> float:
    """

    :param sales:
    :return:
    """
    prices, quantities = zip(*sales)  # Разделение списка на два списка: prices и quantities
    total_price = reduce(lambda acc, x: acc + x, map(lambda p, q: p * q, prices, quantities), 0)
    total_quantity = reduce(lambda acc, q: acc + q, quantities, 0)
    return total_price / total_quantity if total_quantity != 0 else 0


print(weighted_sale_price([(120, 2), (300, 5), (150, 3), (400, 1), (250, 4)]))
