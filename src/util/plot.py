import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def plot_results(predictions, actuals, timestamps, min_vals, max_vals, target_column):
    writer = SummaryWriter()

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    predictions = predictions * (max_vals[target_column] - min_vals[target_column]) + min_vals[target_column]
    actuals = actuals * (max_vals[target_column] - min_vals[target_column]) + min_vals[target_column]
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