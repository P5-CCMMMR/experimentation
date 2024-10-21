import numpy as np

def flex_predict(forecasts, lower_bound, upper_bound, error):
    flex_iter = 0

    for forecast in forecasts:
        if lower_bound + error <= forecast and upper_bound - error >= forecast:    
            flex_iter = flex_iter + 1
        else:
            break
    
    return flex_iter

def prob_flex_predict(forecasts, lower_bound, upper_bound, error):
    flex_iter = 0

    for i in range(0, len(forecasts[0])):
        forecast = np.random.normal(forecasts[0][i], forecasts[1][i]) # 0 being mean and 1 std div
        if lower_bound + error <= forecast and upper_bound - error >= forecast:    
            flex_iter = flex_iter + 1
        else:
            break
    
    return flex_iter