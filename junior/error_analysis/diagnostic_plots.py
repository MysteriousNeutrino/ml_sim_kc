import numpy as np
from scipy.stats import norm

def xy_fitted_residuals(y_true, y_pred):
    """Coordinates (x, y) for fitted residuals against true values."""
    residuals = y_true - y_pred
    return y_pred, residuals

def xy_normal_qq(y_true, y_pred):
    """Coordinates (x, y) for normal Q-Q plot."""
    quantile_levels = np.linspace(0, 1, len(y_true), endpoint=False)
    quantiles = norm.ppf(quantile_levels)

    # Observed quantiles
    residuals = y_true - y_pred
    residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    residuals_sorted = np.sort(residuals)

    return quantiles, residuals_sorted

def xy_scale_location(y_true, y_pred):
    """Coordinates (x, y) for scale-location plot."""
    residuals = y_true - y_pred
    residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    return y_pred, np.sqrt(np.abs(residuals))
