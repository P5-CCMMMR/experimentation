import numpy as np

def create_sequences(data: np.ndarray, seq_len: int):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        x, y = data[i:(i + seq_len)], data[i + seq_len, 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)