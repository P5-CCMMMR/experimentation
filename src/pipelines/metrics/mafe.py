import numpy as np

def mafe(predictions: np.ndarray, actuals: np.ndarray):
    """
    Calculate Mean Absolute Flexibility Error (MAFE) metric
    """
    return np.mean(abs(predictions - actuals))

def maofe(predictions: np.ndarray, actuals: np.ndarray):
    """
    Calculate Mean Absolute Overestimated Flexibility Error (MAOFE) metric
    """
    return np.mean(abs(np.minimum(0, actuals - predictions)))

def maufe(predictions: np.ndarray, actuals: np.ndarray):
    """
    Calculate Mean Absolute Underestimated Flexibility Error (MAUFE) metric
    """
    return np.mean(np.maximum(0, actuals - predictions))

