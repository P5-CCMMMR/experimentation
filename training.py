import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from model import LSTM

def create_sequences(data: np.ndarray, seq_len: int):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        x, y = data[i:(i + seq_len)], data[i + seq_len, 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def trainer(model: nn.Module, epochs: int, data: np.ndarray, learning_rate: float, seq_len: int):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    xs, ys = create_sequences(data, seq_len)
    xs = torch.tensor(xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.float32)

    model.train()

    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(len(xs)):
            inputs = xs[i]
            target = ys[i]

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(xs)
        epoch_losses.append(avg_epoch_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}')
    return epoch_losses
        
# Hyper params
hidden_size = 50
epochs = 10
learning_rate = 0.001
seq_len = 5
model = LSTM(hidden_size)

data = pd.read_csv("NIST_cleaned.csv").iloc[:, 1:].values

epoch_losses = trainer(model, epochs, data, learning_rate, seq_len)

plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.show()
