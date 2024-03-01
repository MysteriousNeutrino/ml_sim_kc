import numpy as np
from scipy.stats import probplot
from scipy.stats import norm

def xy_fitted_residuals(y_true, y_pred):
    """Coordinates (x, y) for fitted residuals against true values."""
    residuals = y_true - y_pred
    return y_pred, residuals

def xy_normal_qq(y_true, y_pred):
    """Coordinates (x, y) for normal Q-Q plot."""
    residuals = y_true - y_pred
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    theoretical_quantiles = norm.ppf(np.linspace(0, 1, len(y_true), endpoint=False))

    sorted_residuals = np.sort(standardized_residuals)

    return theoretical_quantiles, sorted_residuals

def xy_scale_location(y_true, y_pred):
    """Coordinates (x, y) for scale-location plot."""
    residuals = y_true - y_pred
    residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    fitted_vals = y_pred
    sqrt_abs_residuals = np.sqrt(np.abs(residuals))
    return fitted_vals, sqrt_abs_residuals
