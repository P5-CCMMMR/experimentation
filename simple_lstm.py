import torch
import torch.nn as nn

from hyper_parameters import hidden_size, batch_size
from device import device

class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size: int, batch_size: int = 1):
        super(SimpleLSTM, self).__init__()
        self.input_size = 3
        self.output_size = 1
        self.num_layers = 1
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return self.fc(out)

model = SimpleLSTM(hidden_size, batch_size)
model.to(device)