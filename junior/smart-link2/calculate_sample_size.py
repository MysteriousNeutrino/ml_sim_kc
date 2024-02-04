from scipy import stats
import numpy as np


def calculate_sample_size(
        reward_avg: float, reward_std: float, mde: float, alpha: float, beta: float
) -> int:
    """Calculate sample size.

    Parameters
    ----------
    reward_avg: float :
        average reward
    reward_std: float :
        standard deviation of reward
    mde: float :
        minimum detectable effect
    alpha: float :
        significance level
    beta: float :
        type 2 error probability

    Returns
    -------
    int :
        sample size

    """
    assert mde > 0, "mde should be greater than 0"

    z = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(1 - beta)) ** 2
    std = 2 * reward_std ** 2
    effect = (reward_avg * mde) ** 2
    sample_size = (z * std) / effect
    return int(np.round(sample_size, 0))


def calculate_mde(
        reward_std: float, sample_size: int, alpha: float, beta: float
) -> float:
    """Calculate minimal detectable effect.

    Parameters
    ----------
    reward_avg: float :
        average reward
    reward_std: float :
        standard deviation of reward
    sample_size: int :
        sample size
    alpha: float :
        significance level
    beta: float :
        type 2 error probability

    Returns
    -------
    float :
        minimal detectable effect

    """
    z = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(1 - beta))
    std = (2 ** 0.5) * reward_std
    n = sample_size ** 0.5
    mde = (z * std) / n
    return mde
