from src.pipelines.models.model import Model
import torch.nn as nn
import torch

class GRU(Model):
    def __init__(self, hidden_size: int, num_layers: int, input_len: int, horizon_len: int, dropout: float):
        super(GRU, self).__init__(input_len, horizon_len, dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_len, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)
    
    def copy(self):
        return type(self) (
            self.hidden_size, 
            self.num_layers, 
            self.input_len,
            self.output_size, 
            self.dropout
        )
    