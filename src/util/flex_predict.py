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
    
    flexibility = 0
    probabilities = []
    been_out = False
    for mean, stddev in zip(forecasts[0], forecasts[1]):
        within_lower = norm.cdf(lower_bound + error, mean, stddev)
        within_upper = norm.cdf(upper_bound - error, mean, stddev)
        within_bounds = within_upper - within_lower 
        probabilities.append(within_bounds)      
        if within_bounds > confidence and not been_out:
            flexibility += 1
        else:
            been_out = True   
     
    plot_probabilities(probabilities, confidence)
    
    return flexibility, probabilities

def plot_probabilities(probabilities, confidence):
    plt.plot(probabilities, color="b", linestyle="-", marker="o", label="Probability within bounds")
    plt.axhline(y=confidence, color="r", linestyle="--", label=f"Confidence Level ({confidence})")
    plt.xlabel("Time Step")
    plt.xticks(ticks=range(len(probabilities)), labels=range(1, len(probabilities) + 1))
    plt.ylabel("Probability")
    plt.title("Probability of Forecast Falling Within Bounds")
    plt.legend()
    plt.grid(True)
    plt.savefig("probabilities.png")
    