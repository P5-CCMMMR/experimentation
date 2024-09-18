import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from model import model
from hyper_parameters import seq_len, batch_size
from sequenizer import create_sequences
from data import test_data

matplotlib.use("Agg")

def evaluate(model: nn.Module, test_data: np.ndarray, batch_size: int):
    model.eval()
    xs, ys = create_sequences(test_data, seq_len)
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

predictions, actuals = evaluate(model, test_data, batch_size)

rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
mae = np.mean(np.abs(predictions - actuals))
maxe = np.max(np.abs(predictions - actuals))

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAXE: {maxe:.2f}')

plt.plot(predictions, label="Predictions")
plt.plot(actuals, label="Actual")
plt.xlabel("Time")
plt.ylabel("Indoor Temperature")
plt.title("Predictions vs Actual")
plt.legend()
plt.grid()
plt.savefig("model_eval.png")