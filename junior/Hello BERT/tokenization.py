import math
from dataclasses import dataclass
from typing import List, Generator, Tuple

from transformers import PreTrainedTokenizer


@dataclass
class DataLoader:
    """

    """
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128

    def __iter__(self) -> Generator[List[List[int]], None, None]:
        """Iterate over batches"""
        for i in range(len(self)):
            yield self.batch_tokenized(i)

    def __len__(self):
        """Number of batches"""
        with open(self.path, 'r', encoding='utf-8') as file:
            next(file)
            num_lines = sum(1 for line in file)

        return math.ceil(num_lines / self.batch_size)

    def tokenize(self, batch: List[str]) -> List[List[int]]:
        """Tokenize list of texts"""
        batch_vectors = []

        for text in batch:
            vector = self.tokenizer.encode(
                text, max_length=self.max_length, add_special_tokens=True)
            batch_vectors.append(vector)

        return batch_vectors

    def batch_loaded(self, i: int) -> Tuple[List[str], List[int]]:
        """Return loaded i-th batch of data (text, label)"""
        start_index = i * self.batch_size
        end_index = (i + 1) * self.batch_size

        texts = []
        labels = []

        with open(self.path, 'r', encoding='utf-8') as file:
            next(file)
            for index, line in enumerate(file):
                if start_index <= index < end_index:
                    line = line.strip().split(",", 4)
                    sentiment = line[3]
                    label = 1 if sentiment == 'positive' else -1 if sentiment != "neutral" else 0
                    texts.append(line[4])
                    labels.append(label)
                elif index >= end_index:
                    break
        return texts, labels

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """Return tokenized i-th batch of data"""
        texts, labels = self.batch_loaded(i)
        tokens = self.tokenize(texts)
        return tokens, labels
