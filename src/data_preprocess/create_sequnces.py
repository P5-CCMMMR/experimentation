import torch
import numpy as np
import pandas as pd

def create_sequences(features: pd.DataFrame, seq_len: int, time_row_name: str):
    features[time_row_name] = pd.to_datetime(features[time_row_name])
    grouped = features.groupby(features[time_row_name].dt.date)

    xs, ys = [], []
    for _, group in grouped:
        group_features = group.drop(columns=[time_row_name]).values
        num_sequences = len(group_features) - seq_len
        num_features = group_features.shape[1]

        day_xs = np.zeros((num_sequences, seq_len, num_features), dtype=np.float32)
        day_ys = np.zeros((num_sequences, 1), dtype=np.float32)
        for i in range(num_sequences):
            day_xs[i] = group_features[i:i + seq_len]
            day_ys[i] = group_features[i + seq_len, 1]

        xs.append(day_xs)
        ys.append(day_ys)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return torch.tensor(xs), torch.tensor(ys)
