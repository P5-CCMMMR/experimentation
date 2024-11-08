from scipy.stats import norm
import matplotlib.pyplot as plt

def flex_predict(forecasts, lower_bound, upper_bound, error=0):
    flexibility = 0
    for forecast in forecasts:
        if lower_bound + error <= forecast and upper_bound - error >= forecast:    
            flexibility = flexibility + 1
        else:
            break
    
    return flexibility

def prob_flex_predict(forecasts, lower_bound, upper_bound, error=0, confidence=0.95):
    assert confidence > 0 and confidence < 1, "Confidence level must be between 0 and 1"
    assert error >= 0, "Error must be a positive number"
    assert len(forecasts[0]) == len(forecasts[1]), "Mean and standard deviation arrays must be the same length"
    assert lower_bound + error < upper_bound - error, "Lower bound must be less than upper bound"
    
    flexibility = 0
    probabilities = []
    been_out = False
    for mean, stddev in zip(forecasts[0], forecasts[1]):
        within_lower = norm.cdf(lower_bound + error, mean, stddev).item()
        within_upper = norm.cdf(upper_bound - error, mean, stddev).item()
        within_bounds = within_upper - within_lower 
        probabilities.append(within_bounds)      
        if within_bounds > confidence and not been_out:
            flexibility += 1
        else:
            been_out = True   
            
    return flexibility, probabilities
    