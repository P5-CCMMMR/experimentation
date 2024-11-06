import numpy as np
import torch
from scipy.stats import norm

def flex_predict(forecasts, lower_bound, upper_bound, error):
    flex_iter = 0

    for forecast in forecasts:
        if lower_bound + error <= forecast and upper_bound - error >= forecast:    
            flex_iter = flex_iter + 1
        else:
            break
    
    return flex_iter

def prob_flex_predict(forecasts, lower_bound, upper_bound, error, confidence=0.95):
    flex_iter = 0
    
    for mean, stddev in zip(forecasts[0], forecasts[1]):
        prob_within_lower = norm.cdf(lower_bound + error, mean, stddev)
        prob_within_upper = norm.cdf(upper_bound - error, mean, stddev)
        
        prob_within = prob_within_upper - prob_within_lower        
        if prob_within < confidence:
            break
        
        flex_iter += 1   
     
    return flex_iter
