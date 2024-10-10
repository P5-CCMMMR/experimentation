def flex_predict(forecasts, lower_bound, upper_bound, error):
    flex_iter = 0

    for forecast in forecasts:
        if lower_bound + error <= forecast and upper_bound - error >= forecast:    
            flex_iter = flex_iter + 1
        else:
            break
    
    return flex_iter