from scipy.stats import norm

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
    assert lower_bound + error < upper_bound - error, "Lower bound must be less than upper bound"
    
    flexibility = 0
    cumulative_probability = 1
    probabilities = []
    rounding_correction = 1e-16
    
    for mean, stddev in zip(forecasts[0], forecasts[1], strict=True):
        within_lower = norm.cdf(lower_bound + error, mean, stddev).item()
        within_upper = norm.cdf(upper_bound - error, mean, stddev).item()
        within_bounds = within_upper - within_lower
        
        # Probability can never be 1 unless the bounds are infinite, and can never be 0
        if within_bounds == 1 and upper_bound != float("inf") and lower_bound != float("-inf"):
            within_bounds = 1 - rounding_correction
        elif within_bounds == 0:
            within_bounds = rounding_correction
        
        cumulative_probability *= within_bounds
        probabilities.append(cumulative_probability)  
            
        if cumulative_probability >= confidence:
            flexibility += 1 
            
    return flexibility, probabilities
    