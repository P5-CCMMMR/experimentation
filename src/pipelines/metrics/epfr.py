import numpy as np

def epfr(predictions: np.ndarray, actuals: np.ndarray):
    """
    Calculate Extra Predicted Flexibility Ratio (EPFR) metric
    """
    actuals_sum = np.sum(actuals)
    
    if actuals_sum == 0:
        return 0.0
    return np.divide(np.sum(np.maximum(0, actuals - predictions)), actuals_sum)