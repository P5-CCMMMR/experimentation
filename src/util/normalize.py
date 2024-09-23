import numpy as np

def normalize(data: np.ndarray):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals), min_vals, max_vals
