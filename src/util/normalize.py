import numpy as np

def minmax_scale(data: np.ndarray):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals), min_vals, max_vals

def minmax_descale(data: np.ndarray, min_vals, max_vals):
    return data * (max_vals - min_vals) + min_vals
