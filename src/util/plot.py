import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from src.util.constants import TARGET_COLUMN

def plot_results(predictions, actuals, timestamps, min_vals, max_vals):
    if (len(predictions) > 1):
        predictions = ensemble_predictions(predictions)
    else:
        predictions = predictions[0]
    
    if (isinstance(predictions, tuple)):
        plot_probabilistic_results(predictions, actuals, timestamps, min_vals, max_vals)
    else:
        plot_deterministic_results(predictions, actuals, timestamps, min_vals, max_vals)

def plot_probabilistic_results(predictions, actuals, timestamps, min_vals, max_vals):
    writer = SummaryWriter()
        
    mean_predictions, std_predictions = predictions
    mean_predictions = np.array(mean_predictions)
    std_predictions = np.array(std_predictions)
    actuals = np.array(actuals)

    # Rescale predictions and actuals
    mean_predictions = mean_predictions * (max_vals[TARGET_COLUMN] - min_vals[TARGET_COLUMN]) + min_vals[TARGET_COLUMN]
    std_predictions = std_predictions * (max_vals[TARGET_COLUMN] - min_vals[TARGET_COLUMN])
    actuals = actuals * (max_vals[TARGET_COLUMN] - min_vals[TARGET_COLUMN]) + min_vals[TARGET_COLUMN]
    timestamps = timestamps[:len(mean_predictions)]
    
    plt.plot(timestamps, mean_predictions, label="Prediction")
    plt.plot(timestamps, actuals, label="Actual")

    # Uncertainty bands
    plt.fill_between(timestamps, 
                    mean_predictions - 3 * std_predictions, 
                    mean_predictions + 3 * std_predictions, 
                    color='gray', alpha=0.3, label='99.7% (3σ)')
    
    plt.fill_between(timestamps,
                    mean_predictions - 2 * std_predictions,
                    mean_predictions + 2 * std_predictions,
                    color='gray', alpha=0.5, label='95% (2σ)')
    
    plt.fill_between(timestamps,
                    mean_predictions - std_predictions,
                    mean_predictions + std_predictions,
                    color='gray', alpha=0.8, label='68% (σ)')

    plt.xlabel("Time")
    plt.ylabel("Indoor Temperature")
    plt.title("Predictions vs Actuals with Uncertainty")
    plt.legend()
    plt.grid()
    plt.gcf().autofmt_xdate()

    # Save plot to TensorBoard
    writer.add_figure("predictions", plt.gcf())
    writer.close()

def plot_deterministic_results(predictions, actuals, timestamps, min_vals, max_vals):
    writer = SummaryWriter()

    predictions = np.array(predictions)    
    actuals = np.array(actuals)

    # Rescale predictions and actuals 
    predictions = predictions * (max_vals[TARGET_COLUMN] - min_vals[TARGET_COLUMN]) + min_vals[TARGET_COLUMN]
    actuals = actuals * (max_vals[TARGET_COLUMN] - min_vals[TARGET_COLUMN]) + min_vals[TARGET_COLUMN]
    timestamps = timestamps[:len(predictions)]

    plt.plot(timestamps, predictions, label="Prediction")
    plt.plot(timestamps, actuals, label="Actual")

    plt.xlabel("Time")
    plt.ylabel("Indoor Temperature")
    plt.title("Predictions vs Actuals")
    plt.legend()
    plt.grid()
    plt.gcf().autofmt_xdate()
    
    writer.add_figure("predictions", plt.gcf())
    writer.close()
    
def ensemble_predictions(predictions):
    if (isinstance(predictions[0], tuple)):
        return ensemble_probabilistic_predictions(predictions)
    return ensemble_deterministic_predictions(predictions)

def ensemble_probabilistic_predictions(predictions):
    mean_predictions = []
    std_predictions = []
    
    for i in range(len(predictions[0][0])):
        mean_row = []
        std_row = []
        
        for j in range(len(predictions)):
            mean_row.append(predictions[j][0][i])
            std_row.append(predictions[j][1][i])
    
        mean_mixture = np.mean(mean_row)
        std_mixture = np.sqrt(np.sum([n**2 for n in std_row] + [n**2 for n in mean_row]) / len(std_row) - mean_mixture**2)
        
        mean_predictions.append(mean_mixture)
        std_predictions.append(std_mixture)
        
    return mean_predictions, std_predictions

def ensemble_deterministic_predictions(predictions):
    mean_predictions = []
    std_predictions = []

    for i in range(len(predictions[0])):
        row = []
        for j in range(len(predictions)):
            row.append(predictions[j][i])
            
        mean_prediction = np.mean(row)
        std_prediction = np.std(row)
        
        mean_predictions.append(mean_prediction)
        std_predictions.append(std_prediction)
        
    return mean_predictions, std_predictions
