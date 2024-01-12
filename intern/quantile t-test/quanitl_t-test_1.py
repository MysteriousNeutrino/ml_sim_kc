from typing import List, Tuple
from scipy.stats import mannwhitneyu


def ttest(
        control: List[float],
        experiment: List[float],
        alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Two-sample t-test for the means of two independent samples"""
    _, p_value = mannwhitneyu(control, experiment, alternative='less')

    return p_value, bool(p_value < alpha)
