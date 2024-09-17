import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from model import LSTM
from hyper_parameters import hidden_size, epochs, learning_rate, seq_len
from sequenizer import create_sequences
from data import train_data

matplotlib.use("Agg")

def trainer(model: nn.Module, epochs: int, train_data: np.ndarray, learning_rate: float, seq_len: int):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    xs, ys = create_sequences(train_data, seq_len)
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

    torch.save(model.state_dict(), "trained_model.pth")
    return epoch_losses
        
model = LSTM(hidden_size)

epoch_losses = trainer(model, epochs, train_data, learning_rate, seq_len)

plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid()
plt.savefig("training_loss.png")
