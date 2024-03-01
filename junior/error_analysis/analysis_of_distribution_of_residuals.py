from typing import Tuple, Optional
import numpy as np
from scipy.stats import shapiro, ttest_1samp, levene, fligner, bartlett


def test_normality(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float = 0.05
) -> Tuple[float, bool]:
    """Normality test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the normality test

    is_rejected : bool
        True if the normality hypothesis is rejected, False otherwise

    """
    _, p_value = shapiro((y_true - y_pred))

    return p_value, p_value < alpha


def test_unbiased(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefer: Optional[str] = None,
        alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Unbiasedness test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    prefer : str, optional (default=None)
        If None or "two-sided", test whether the residuals are unbiased.
        If "positive", test whether the residuals are unbiased or positive.
        If "negative", test whether the residuals are unbiased or negative.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the unbiasedness hypothesis is rejected, False otherwise

    """
    if prefer == "positive":
        alternative = "greater"
    elif prefer == "negative":
        alternative = "less"
    else:
        alternative = "two-sided"

    _, p_value = ttest_1samp((y_true - y_pred), popmean=0.0, alternative=alternative)

    return p_value, p_value < alpha


def test_homoscedasticity(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bins: int = 10,
        test: Optional[str] = None,
        alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Homoscedasticity test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    bins : int, optional (default=10)
        Number of bins to use for the test.
        All bins are equal-width and have the same number of samples, except
        the last bin, which will include the remainder of the samples
        if n_samples is not divisible by bins parameter.

    test : str, optional (default=None)
        If None or "bartlett", perform Bartlett's test for equal variances.
        If "levene", perform Levene's test.
        If "fligner", perform Fligner-Killeen's test.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the homoscedasticity hypothesis is rejected, False otherwise

    """
    # sort remainders
    # bins
    # levene
    remainders = y_true - y_pred
    sort_remainders = remainders[np.argsort(y_true)]
    num_bins = len(sort_remainders) // bins
    # след 5 строк можно заменить на 1
    # chunks = [sort_remainders[i : i + num_bins] for i in range(0, len(sort_remainders), num_bins)]
    first_part_data = sort_remainders[:bins * num_bins]
    subarrays = list(np.split(first_part_data, bins))
    if len(remainders) % bins != 0:
        second_part_data = sort_remainders[bins * num_bins:]
        subarrays.append(second_part_data)

    if test == "levene":
        _, p_value = levene(*subarrays)
    elif test == "fligner":
        _, p_value = fligner(*subarrays)
    elif test == "bartlett" or test is None:
        _, p_value = bartlett(*subarrays)
    else:
        raise ValueError
    return p_value, p_value < alpha
