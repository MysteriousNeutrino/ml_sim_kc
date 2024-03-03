from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class ImageKMeans:
    n_clusters: int = 5
    init: str | np.ndarray = "random"
    max_iter: int = 100
    random_state: int = 42

    def _image_as_array(self, image: np.ndarray) -> np.ndarray:
        """Convert image to pixel array"""
        X = np.array(image.reshape(-1, 3))
        return X

    def _init_centroids(self, X: np.ndarray) -> None:
        """Select N random samples as initial centroids"""
        np.random.seed(self.random_state)
        if isinstance(self.init, str) and self.init == "random":
            centroid_indexes = np.random.choice(np.arange(len(X)), size=self.n_clusters, replace=False)
            self.centroids_ = X[centroid_indexes]
        elif isinstance(self.init, str):
            raise ValueError("неправильная строка")
        elif isinstance(self.init, np.ndarray) and self.init.shape == (self.n_clusters, 3) and (
                self.init >= 0).all() and (self.init <= 255).all() and len(np.unique(self.init, axis=0)) ==len(self.init):
            self.centroids_ = self.init
        else:
            raise TypeError("неправильный массив")

    def _assign_centroids(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to the closest centroid"""
        y = np.zeros(len(X))

        for i in range(len(X)):
            ...

        return y


image = ImageKMeans(init = np.array([[186,214,246],
[240,95,193],
[206,228,69],
[86,228,234],
[178,52,227]]))

image._init_centroids(image._image_as_array(np.random.randint(0, 256, size=(1080, 1920, 3), dtype=np.uint8)))
