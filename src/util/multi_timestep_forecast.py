from src.data_preprocess.usage_timeseries_dataset import UsageTimeSeriesDataset
from src.util.normalize import minmax_scale, minmax_descale
from torch.utils.data import DataLoader
import numpy as np

def multiTimestepForecasting(model, data, sequence_len):
    in_temp_idx = 1
    seq, min_vals, max_vals = minmax_scale(data[:, 1:].astype(float)) # exclude first column

    min_in_temp = min_vals[in_temp_idx:in_temp_idx + 1]
    max_in_temp = max_vals[in_temp_idx:in_temp_idx + 1]

    dataset = UsageTimeSeriesDataset(seq, sequence_len)
    dataloader = DataLoader(dataset, batch_size=1)

    batch = next(iter(dataloader))  # only run 1 time but i don't know how else to get a batch out of dataloader 
    outputs = model(batch)
    outputs = outputs.detach()

    return minmax_descale(outputs, min_in_temp , max_in_temp)

