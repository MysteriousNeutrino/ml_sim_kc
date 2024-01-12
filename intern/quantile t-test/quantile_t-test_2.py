from typing import List
from typing import Tuple

import numpy as np
from scipy.stats import ttest_ind


def quantile_ttest(
        control: List[float],
        experiment: List[float],
        alpha: float = 0.05,
        quantile: float = 0.95,
        n_bootstraps: int = 1000,
) -> Tuple[float, bool]:
    """
    Bootstrapped t-test for quantiles of two samples.
    """
    control_95_quantile = []
    experiment_95_quantile = []

    for _ in range(n_bootstraps):
        bootstrapped_control = np.random.choice(control, size=len(control), replace=True)
        bootstrapped_experiment = np.random.choice(experiment, size=len(experiment), replace=True)

        control_95_quantile.append(np.quantile(bootstrapped_control, q=quantile))
        experiment_95_quantile.append(np.quantile(bootstrapped_experiment, q=quantile))

    _, p_value = ttest_ind(control_95_quantile, experiment_95_quantile, equal_var=True)
    return p_value, bool(p_value < alpha)
