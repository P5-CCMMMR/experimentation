import torch
import torch.nn as nn
import pandas as pd

data = pd.read_csv("HVAC-hour_cleaned.csv").values

class TestModel(nn.Module):
    def __init__(self, stacked_layers: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(stacked_layers, hidden_size, stacked_layers)
        self.label = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[-1, :]
        return self.fc(out).squeeze()
    