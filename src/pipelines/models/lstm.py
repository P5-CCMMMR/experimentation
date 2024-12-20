from .model import Model
import torch.nn as nn
import torch

class LSTM(Model):
    def __init__(self, hidden_size: int, num_layers: int, input_len: int, horizon_len: int, dropout: float):
        super(LSTM, self).__init__(input_len, horizon_len, dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        if num_layers == 1:
            self.lstm = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True, dropout=dropout)    
    
        self.dropout_module = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout_module(out[:, -1, :])
        return self.fc(out)
    
    def copy(self):
        return type(self) (
            self.hidden_size, 
            self.num_layers, 
            self.input_len,
            self.output_size, 
            self.dropout
        )
    