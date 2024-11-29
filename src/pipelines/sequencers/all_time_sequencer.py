import torch
from .sequencer import Sequencer

class AllTimeSequencer(Sequencer):
    def __len__(self):
        return max(0, len(self.data) - (self.seq_len + self.horizon_len)) 

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len, :]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.horizon_len, self.target_column]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
