from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    # YOUR CODE HERE: add the required attributes
    threshold: float = None
    n_samples: int = None
    value: int = None
    mse: int = None
    left: Node = None
    right: Node = None
