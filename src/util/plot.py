import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def plot_results(predictions, actuals, timestamps, min_vals, max_vals, target_column):
    writer = SummaryWriter()

    mean_predictions, std_predictions = predictions
    mean_predictions = np.array(mean_predictions)
    std_predictions = np.array(std_predictions)
    actuals = np.array(actuals)

    # Rescale predictions and actuals
    mean_predictions = mean_predictions * (max_vals[target_column] - min_vals[target_column]) + min_vals[target_column]
    std_predictions = std_predictions * (max_vals[target_column] - min_vals[target_column])
    actuals = actuals * (max_vals[target_column] - min_vals[target_column]) + min_vals[target_column]
    timestamps = timestamps[:len(mean_predictions)]

    plt.plot(timestamps, mean_predictions, label="Prediction")
    plt.plot(timestamps, actuals, label="Actual")

    # Uncertainty bands
    plt.fill_between(timestamps, 
                     mean_predictions - 3 * std_predictions, 
                     mean_predictions + 3 * std_predictions, 
                     color='gray', alpha=0.2, label='3σ')
    
    plt.fill_between(timestamps,
                     mean_predictions - 2 * std_predictions,
                     mean_predictions + 2 * std_predictions,
                     color='gray', alpha=0.5, label='2σ')
    
    plt.fill_between(timestamps,
                     mean_predictions - std_predictions,
                     mean_predictions + std_predictions,
                     color='gray', alpha=0.8, label='σ')

    plt.xlabel("Time")
    plt.ylabel("Indoor Temperature")
    plt.title("Predictions vs Actuals with Uncertainty")
    plt.legend()
    plt.grid()
    plt.gcf().autofmt_xdate()

    # Save plot to TensorBoard
    writer.add_figure("predictions", plt.gcf())
    writer.close()