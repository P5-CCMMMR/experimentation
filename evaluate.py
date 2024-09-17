import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from model import LSTM
from hyper_parameters import hidden_size, seq_len
from sequenizer import create_sequences
from data import test_data
from device import device

matplotlib.use("Agg")

def evaluate(model: nn.Module, test_data: np.ndarray, device: torch.device):
    model.to(device)
    model.eval()
    xs, ys = create_sequences(test_data, seq_len)
    xs = torch.tensor(xs, dtype=torch.float32).to(device)
    ys = torch.tensor(ys, dtype=torch.float32).to(device)

    predictions = []
    with torch.no_grad():
        for i in range(len(xs)):
            input_seq = xs[i]
            prediction = model(input_seq)
            predictions.append(prediction.item())
    
    return np.array(predictions), ys.numpy()
    
model = LSTM(hidden_size)
model.load_state_dict(torch.load("trained_model.pth", weights_only=True))

predictions, actuals = evaluate(model, test_data)

mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
print(f'Total Percentage Error (MAPE): {mape:.2f}%')

plt.plot(predictions, label="Predictions")
plt.plot(actuals, label="Actual")
plt.xlabel("Time")
plt.ylabel("Indoor Temperature")
plt.title("Predictions vs Actual")
plt.legend()
plt.grid()
plt.savefig("model_eval.png")