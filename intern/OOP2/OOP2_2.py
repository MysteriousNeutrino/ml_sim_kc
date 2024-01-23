from abc import ABC, abstractmethod


class DataPreprocessor(ABC):
    """
    абстрактный класс для пайплайна предобработки данных
    """

    @abstractmethod
    def preprocess(self, data):
        """

        :param data:
        :return:
        """
        pass


class OutlierRemover(DataPreprocessor):
    """
    OutlierRemover
    """

    def preprocess(self, data):
        return [x for x in data if x <= 10]


class Normalizer(DataPreprocessor):
    """
    Normalizer
    """

    def preprocess(self, data):
        return [x / 10 for x in data]


class Encoder(DataPreprocessor):
    """
    Encoder
    """

    def __init__(self, encoding_dict: dict):
        self.encoding_dict = encoding_dict

    def preprocess(self, data):
        """

        :param data:
        :return:
        """
        return [self.encoding_dict.get(x, 0) for x in data]

# Пример использования OutlierRemover
outlier_remover = OutlierRemover()

data_with_outliers = [1, 2, 3, 100, 5, 6, 7, 8, 9]
cleaned_data = outlier_remover.preprocess(data_with_outliers)

print(f"Исходные данные: {data_with_outliers}")
print(f"Данные без выбросов: {cleaned_data}")


# Пример использования Normalizer
normalizer = Normalizer()

numerical_data = [10, 20, 30, 40, 50]
normalized_data = normalizer.preprocess(numerical_data)

print(f"Исходные числовые данные: {numerical_data}")
print(f"Нормализованные данные: {normalized_data}")

# Пример использования Encoder
encoder = Encoder(encoding_dict={'зеленый': 2, 'синий': 3})

categorical_data = ['красный', 'зеленый', 'синий']
encoded_data = encoder.preprocess(categorical_data)

print(f"Исходные категориальные данные: {categorical_data}")
print(f"Закодированные данные: {encoded_data}")

