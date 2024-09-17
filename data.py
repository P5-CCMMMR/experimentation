import pandas as pd

data = pd.read_csv("NIST_cleaned.csv").iloc[:, 1:].values
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]