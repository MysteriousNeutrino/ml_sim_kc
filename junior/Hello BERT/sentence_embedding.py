import math
from dataclasses import dataclass
from typing import List, Generator, Tuple

import torch
from transformers import PreTrainedTokenizer, DistilBertTokenizer, DistilBertModel


@dataclass
class DataLoader:
    """

    """
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128
    padding: str = None

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

    def paddding(self, texts):
        if self.padding == None:
            return texts
        elif self.padding == 'max_length':
            padded_sequences = []
            for text in texts:
                text = text + [0] * (self.max_length - len(text))
                padded_sequences.append(text)
            return padded_sequences
        elif self.padding == 'batch':
            padded_sequences = []
            max_length = max(len(text) for text in texts)
            for text in texts:
                text = text + [0] * (max_length - len(text))
                padded_sequences.append(text)
            return padded_sequences

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """Return tokenized i-th batch of data"""
        texts, labels = self.batch_loaded(i)
        tokens = self.paddding(self.tokenize(texts))
        return tokens, labels


def attention_mask(padded: List[List[int]]) -> List[List[int]]:
    masks = []
    for text in padded:
        mask = [0 if char == 0 else 1 for char in text]
        masks.append(mask)
    return masks

def review_embedding(tokens: List[List[int]], model) -> List[List[float]]:
    """Return embedding for batch of tokenized texts"""
    mask = attention_mask(tokens)

    # Calculate embeddings
    tokens = torch.tensor(tokens)
    mask = torch.tensor(mask)

    with torch.no_grad():
        last_hidden_states = model(tokens, attention_mask=mask)

    # Embeddings for [CLS]-tokens
    features = last_hidden_states[0][:, 0, :].tolist()
    return features


# MODEL_NAME = 'distilbert-base-uncased'
# tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
# bert = DistilBertModel.from_pretrained(MODEL_NAME)
#
# loader = DataLoader('./data/reviews.csv', tokenizer, max_length=128, padding='batch')
#
# for tokens, labels in loader:
#     review_embedding(tokens, bert)


