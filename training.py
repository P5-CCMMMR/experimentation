import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from simple_lstm import model
from hyper_parameters import epochs, learning_rate, seq_len, batch_size
from sequenizer import create_sequences
from data import train_data
from device import device

matplotlib.use("Agg")

def trainer(model: nn.Module, epochs: int, train_data: np.ndarray, learning_rate: float, seq_len: int, device: torch.device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    xs, ys = create_sequences(train_data, seq_len)
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

epoch_losses = trainer(model, epochs, train_data, learning_rate, seq_len, device)

torch.save(model.state_dict(), "trained_model.pth")

plt.plot(range(1, epochs + 1), epoch_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Training Loss Over Epochs")
plt.grid()
plt.savefig("training_loss.png")
