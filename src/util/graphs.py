
import matplotlib.pyplot as plt
import pandas as pd

def create_eval_graph(): 
    timestamps = pd.to_datetime(timestamps)

    # Make all same size
    min_length = min(len(timestamps), len(predictions), len(actuals))
    timestamps = timestamps[:min_length]
    predictions = predictions[:min_length]
    actuals = actuals[:min_length]

    plt.plot(timestamps, predictions, label="Predictions")
    plt.plot(timestamps, actuals, label="Actual")
    plt.xlabel("Time")
    plt.ylabel("Indoor Temperature")
    plt.title("Predictions vs Actual")
    plt.legend()
    plt.grid()
    plt.gcf().autofmt_xdate()
    plt.savefig(GRAPH_PATH + "model_eval.png")