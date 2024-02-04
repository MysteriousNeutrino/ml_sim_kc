from typing import Tuple

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
    for i in range(n_simulations):
        sample1 = cpc_sample(n_samples, conversion_rate, reward_avg, reward_std)
        sample2 = cpc_sample(n_samples, conversion_rate * (1+ mde), reward_avg, reward_std)
        if not t_test(sample1, sample2, alpha)[0]:
            counter += 1

    return counter / n_simulations


# print(ab_test(1000, 5000, 0.60, 0.05, 20, 1, 0.05))
