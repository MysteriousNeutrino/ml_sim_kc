import torch


def contrastive_loss(
        x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor, margin: float = 5.0
) -> torch.Tensor:
    """
    Computes the contrastive loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        x1 (torch.Tensor): Embedding vectors of the
            first objects in the pair (shape: (N, M))
        x2 (torch.Tensor): Embedding vectors of the
            second objects in the pair (shape: (N, M))
        y (torch.Tensor): Ground truth labels (1 for similar, 0 for dissimilar)
            (shape: (N,))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The contrastive loss
    """
    y = torch.clone(y)
    point_distance = torch.norm(x1 - x2, p=2, dim=1)
    # print("point_distance ^ 2:", (torch.tensor(margin) - point_distance) ** 2)
    # print("point_distance:", margin-point_distance)
    L = torch.where(y.bool(), point_distance ** 2,
                    torch.max(torch.tensor(margin) - point_distance, torch.zeros_like(point_distance)) ** 2)
    # print(L, len(L))
    return torch.mean(L)

# print(
# contrastive_loss(x1=torch.tensor([[-0.8600, -8.0300], [-2.7800, -0.6500]], dtype=torch.float64),
#                  x2=torch.tensor([[-8.7200, -4.0200], [-0.0500, 4.2400]], dtype=torch.float64),
#                  y=torch.tensor([1, 0]), margin=13.0))
