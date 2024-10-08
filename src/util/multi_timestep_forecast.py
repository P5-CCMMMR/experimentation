from src.data_preprocess.usage_timeseries_dataset import UsageTimeSeriesDataset
from src.util.normalize import normalize, denormalize
from torch.utils.data import DataLoader
import numpy as np

def multiTimestepForecasting(model, data, sequence_len):
    out_temp_idx = 2
    in_temp_idx = 1
    power_idx = 0
    last_input_idx = sequence_len - 1

    predictions = []
    seq, min_vals, max_vals = normalize(data[:, 1:].astype(float)) # exclude first column

    last_out_temp = seq[last_input_idx][out_temp_idx] 
    last_power = seq[last_input_idx][power_idx]
    
    min_in_temp = min_vals[in_temp_idx:in_temp_idx + 1]
    max_in_temp = max_vals[in_temp_idx:in_temp_idx + 1]

    for _ in range(0, sequence_len):
        dataset = UsageTimeSeriesDataset(seq, sequence_len)
        dataloader = DataLoader(dataset, batch_size=1)

        batch = next(iter(dataloader))  # only run 1 time but i don't know how else to get a batch out of dataloader 
        outputs = model(batch)
        predictions.append(outputs.item())

        seq = seq[1:sequence_len]
        new_row = np.array([[last_power, predictions[len(predictions) - 1], last_out_temp]])
        seq = np.append(seq, new_row, axis=0)

    return denormalize(predictions, min_in_temp , max_in_temp)

