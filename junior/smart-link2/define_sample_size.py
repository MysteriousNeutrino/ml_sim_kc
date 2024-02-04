from typing import List, Tuple

import numpy as np
from scipy.stats import ttest_ind


def cpc_sample(
        n_samples: int, conversion_rate: float, reward_avg: float, reward_std: float
) -> np.ndarray:
    """Sample data."""

    CRV = np.random.binomial(1, conversion_rate, n_samples)
    CPA = np.random.normal(reward_avg, reward_std, n_samples)
    return CRV * CPA


def t_test(cpc_a: np.ndarray, cpc_b: np.ndarray, alpha=0.05
           ) -> Tuple[bool, float]:
    """Perform t-test.

    Parameters
    ----------
    cpc_a: np.ndarray :
        first samples
    cpc_b: np.ndarray :
        second samples
    alpha :
         (Default value = 0.05)

    Returns
    -------
    Tuple[bool, float] :
        True if difference is significant, False otherwise
        p-value
    """

    t_test = ttest_ind(cpc_a, cpc_b)

    return (bool(t_test.pvalue < alpha), t_test.pvalue)


def aa_test(
        n_simulations: int,
        n_samples: int,
        conversion_rate: float,
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
) -> float:
    """Do the A/A test (simulation)."""

    counter = 0
    for _ in range(n_simulations):
        sample1 = cpc_sample(n_samples, conversion_rate, reward_avg, reward_std)
        sample2 = cpc_sample(n_samples, conversion_rate, reward_avg, reward_std)
        if t_test(sample1, sample2, alpha)[0]:
            counter += 1

    return counter / n_simulations


def ab_test(
        n_simulations: int,
        n_samples: int,
        conversion_rate: float,
        mde: float,
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
) -> float:
    """Do the A/B test (simulation)."""

    counter = 0
    for _ in range(n_simulations):
        sample1 = cpc_sample(n_samples, conversion_rate, reward_avg, reward_std)
        sample2 = cpc_sample(n_samples, conversion_rate * (1 + mde), reward_avg, reward_std)
        if not t_test(sample1, sample2, alpha)[0]:
            counter += 1

    return counter / n_simulations


def select_sample_size(
        n_samples_grid: List[int],
        n_simulations: int,
        conversion_rate: float,
        mde: float,
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
        beta: float = 0.2,
) -> Tuple[int, float, float]:
    """Select sample size."""
    for n_samples in n_samples_grid:
        type_1_error = aa_test(n_simulations, n_samples,
                               conversion_rate, reward_avg, reward_std, alpha)
        type_2_error = ab_test(n_simulations, n_samples,
                               conversion_rate, mde, reward_avg, reward_std, alpha)

        if type_1_error <= alpha and type_2_error <= beta:
            return n_samples, type_1_error, type_2_error

    raise RuntimeError(
        "Can't find sample size. "
        f"Last sample size: {n_samples}, "
        f"last type 1 error: {type_1_error}, "
        f"last type 2 error: {type_2_error}"
        "Make sure that the grid is big enough."
    )


def select_mde(
        n_samples: int,
        n_simulations: int,
        conversion_rate: float,
        mde_grid: List[float],
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
        beta: float = 0.2,
) -> Tuple[float, float]:
    """Select MDE."""
    for mde in mde_grid:
        type_2_error = ab_test(n_simulations, n_samples,
                               conversion_rate, mde, reward_avg, reward_std, alpha)

        if type_2_error <= beta:
            return mde, type_2_error

    raise RuntimeError(
        "Can't find MDE. "
        f"Last MDE: {mde}, "
        f"last type 2 error: {type_2_error}. "
        "Make sure that the grid is big enough."
    )

# print(select_sample_size([100, 1000, 10000], 50000, 0.60, 0.05, 20, 1, 0.05, 0.2))
