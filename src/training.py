import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from src.util.simple_lstm import model
from src.util.hyper_parameters import GRAPH_PATH, epochs, learning_rate, seq_len, batch_size
from src.util.sequenizer import create_sequences
from src.data_preprocess.data import train_data
from src.util.device import device
from src.evaluate import evaluate
from .data_preprocess.data import test_data
import numpy as np

matplotlib.use("Agg")

def trainer(model: nn.Module, epochs: int, features: pd.DataFrame, learning_rate: float, seq_len: int):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    xs, ys = create_sequences(features, seq_len)
    loader = data.DataLoader(data.TensorDataset(xs, ys), batch_size=batch_size, drop_last=True, shuffle=True)

    model.to(device)
    model.train()

    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % batch_size == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(loader) - 1}], Loss: {loss.item():.4f}')
        epoch_losses.append(epoch_loss)
    torch.save(model.state_dict(), "trained_model.pth")
    return epoch_losses

best_rmse = None

for i in range(0, 1): 
    epoch_losses = trainer(model, epochs, train_data, learning_rate, seq_len)

    predictions, actuals = evaluate(model, test_data, batch_size)

    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    maxe = np.max(np.abs(predictions - actuals))
    
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'MAXE: {maxe:.2f}')

    if (best_rmse == None or rmse < best_rmse):
        best_rmse = rmse
        torch.save(model.state_dict(), "trained_model.pth")

plt.plot(range(1, epochs + 1), epoch_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Training Loss Over Epochs")
plt.grid()
plt.savefig(GRAPH_PATH + "/training_loss.png")

timestamps = test_data.values[:, 0]
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
    