import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
from .util.simple_lstm import model
from .util.hyper_parameters import GRAPH_PATH, seq_len, batch_size
from .util.sequenizer import create_sequences
from .data_preprocess.data import test_data

matplotlib.use("Agg")

def evaluate(model: nn.Module, features: pd.DataFrame, batch_size: int):
    model.eval()
    xs, ys = create_sequences(features, seq_len)
    loader = data.DataLoader(data.TensorDataset(xs, ys), batch_size=batch_size, drop_last=True)

    predictions = []
    actuals = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            prediction = model(x_batch)
            predictions.append(prediction.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    return np.array(predictions), np.array(actuals)
    
model.load_state_dict(torch.load("trained_model.pth", weights_only=True))

timestamps = test_data.values[:, 0]

predictions, actuals = evaluate(model, test_data, batch_size)

rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
mae = np.mean(np.abs(predictions - actuals))
maxe = np.max(np.abs(predictions - actuals))

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAXE: {maxe:.2f}')

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