import torch
import numpy as np
import pandas as pd
from src.util.device import device
from src.util.hyper_parameters import TIMESTAMP

def create_sequences(features: pd.DataFrame, seq_len: int):
    """
    Create sequences of features and corresponding labels from a DataFrame.

    Args:
        features (pd.DataFrame): DataFrame containing the features with a TIMESTAMP column.
        seq_len (int): Length of the sequences to be created.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing the sequences of features (xs) 
                                           and the corresponding labels (ys).
    """
    # Ensure TIMESTAMP is in datetime format
    features[TIMESTAMP] = pd.to_datetime(features[TIMESTAMP])

    # Split data by day
    grouped = features.groupby(features[TIMESTAMP].dt.date)

    xs, ys = [], []
    for _, group in grouped:
        # Drop the TIMESTAMP column and convert the remaining data to a numpy array
        group_features = group.drop(columns=[TIMESTAMP]).values
        # Calculate the number of sequences that can be created from the group
        num_sequences = len(group_features) - seq_len
        # Get the number of features in the data
        num_features = group_features.shape[1]

        # Initialize arrays to hold the sequences and labels for the current day
        day_xs = np.zeros((num_sequences, seq_len, num_features), dtype=np.float32)
        day_ys = np.zeros((num_sequences, 1), dtype=np.float32)
        for i in range(num_sequences):
            # Create a sequence of length seq_len
            day_xs[i] = group_features[i:i + seq_len]
            # The label for the sequence is the value of the second feature at the end of the sequence
            day_ys[i] = group_features[i + seq_len, 1]

        # Append the sequences and labels for the current day to the lists
        xs.append(day_xs)
        ys.append(day_ys)

    # Combine sequences from all days
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    # Convert the sequences and labels to torch tensors and move them to the specified device
    return torch.tensor(xs).to(device), torch.tensor(ys).to(device)
    