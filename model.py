import torch.nn as nn
import pandas as pd

data = pd.read_csv("HVAC-hour_cleaned.csv").values

class TestModel(nn.Module):
    def __init__(self, stacked_layers: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(stacked_layers, hidden_size, stacked_layers)
        self.label = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (hf, cf) = self.lstm(data)
        return self.label(out[-1])

model = TestModel(1, 2)

print(model)

def trainer(model, epoch):
    model.train()

    for i in range(epoch): 
        for j in range(len(data)):
            pass
            

trainer(model, 10)