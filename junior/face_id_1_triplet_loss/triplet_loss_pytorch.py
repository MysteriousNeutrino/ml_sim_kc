import torch


def triplet_loss(
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        margin: float = 5.0,
) -> torch.Tensor:
    """
    Computes the triplet loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        anchor (torch.Tensor): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (torch.Tensor): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (torch.Tensor): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The triplet loss
    """
    point_distance_positive = torch.norm(anchor - positive, p=2, dim=1)
    point_distance_negative = torch.norm(anchor - negative, p=2, dim=1)

    L = torch.maximum(
        torch.tensor(0), point_distance_positive - point_distance_negative + torch.tensor(margin))

    return torch.mean(L)
