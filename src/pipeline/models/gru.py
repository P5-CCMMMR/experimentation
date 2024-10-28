from model import Model
import torch.nn as nn
import torch

class GRU(Model):
    def __init__(self, hidden_size: int, num_layers: int, input_len: int, horison_len: int, dropout: float):
        super(GRU, self).__init__(hidden_size, num_layers, input_len, horison_len, dropout)
        self.gru = nn.GRU(self.input_len, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        return self.fc(out).squeeze()