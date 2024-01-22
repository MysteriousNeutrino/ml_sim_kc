from typing import List, Union

import numpy as np


class DataProcessor:
    """
    to prepared data
    """

    def __init__(self, data):
        self.data: List[int] = data
        self.processed_data_: Union[None, List[int]] = None

    def process(self):
        """

        :return:
        """
        if len(self.data) > 0 and self.data is not None:
            data_mean = np.mean(self.data)
            print("data_mean: ", data_mean)
            self.processed_data_ = list(np.array(self.data) - data_mean)
        return self

    def save_to_file(self, file_name):
        """

        :param file_name:
        :return:
        """
        if self.processed_data_ is not None:
            with open(f"{file_name}", "w", encoding='utf-8') as f:
                for item in self.processed_data_:
                    f.write(str(item) + "\n")
        return self


# Example of usage
processor = DataProcessor(data=[1, 2, 3, 4, 5])
processor.process()
processor.save_to_file("processed_data.txt")
