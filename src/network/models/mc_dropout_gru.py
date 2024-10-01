import torch
import torch.nn as nn

class MCDropoutGRU(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super(MCDropoutGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        input_size = 3
        output_size = 1
        self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        return self.fc(out).squeeze()
    