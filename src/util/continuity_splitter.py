import numpy as np
import pandas as pd

def split_dataframe_by_continuity(df, time_difference: int, sequence_min_len: int, timestamp_col: int):
    sequences = []
    temp_sequence = []
    last_time = None
    for _, row in df.iterrows():
        if last_time is not None and pd.to_datetime(row[timestamp_col]) - last_time != pd.Timedelta(minutes=time_difference):
            if len(temp_sequence) > sequence_min_len:
                sequences.append(np.array(temp_sequence))
            temp_sequence = []
        temp_sequence.append(row)
        last_time = pd.to_datetime(row[timestamp_col])
    if temp_sequence:
        sequences.append(np.array(temp_sequence))
    return sequences