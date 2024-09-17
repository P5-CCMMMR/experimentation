import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, hidden_size: int):
        super(LSTM, self).__init__()
        self.output_size = 1
        self.num_layers = 1
        self.input_size = 3
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[-1, :]
        return self.fc(out).squeeze()