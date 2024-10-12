import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, horizon_len: int, target_column: int):
        self.data = data
        self.horizon_len = horizon_len
        self.target_column = target_column

    def __len__(self):
        return len(self.data) - (self.horizon_len * 2)

    def __getitem__(self, idx):
        x_end = idx + self.horizon_len
        x = self.data[idx:x_end, :]
        y = self.data[x_end: x_end + self.horizon_len, self.target_column]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
