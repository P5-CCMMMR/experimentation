import torch
from src.data_preprocess.timeseries_dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from src.util.normalize import minmax_scale
from src.util.error import RMSE
from src.util.constants import TARGET_COLUMN
from src.util.flex_predict import flex_predict, prob_flex_predict

def get_prob_mafe(data_arr, model, seq_len, error, boundary, time_horizon):
    flex_predictions = []
    flex_actual_values = []

    for data in data_arr:
        if len(data) < seq_len:
            continue

        data, _, _ = minmax_scale(data[:, 1:].astype(float))

        dataset = TimeSeriesDataset(data, seq_len, time_horizon, TARGET_COLUMN)
        dataloader = DataLoader(dataset, 1)

        for batch in dataloader:
            input_data, result_actual = batch  

            last_in_temp  = input_data[:, -1, 2:]

            lower_boundery = last_in_temp - boundary
            upper_boundery = last_in_temp + boundary

            result_predictions = model(input_data)

            actual_flex = flex_predict(result_actual[0], lower_boundery, upper_boundery, error)
            predicted_flex = prob_flex_predict(result_predictions, lower_boundery, upper_boundery, error)

            flex_predictions.append(predicted_flex)
            flex_actual_values.append(actual_flex)
        
    flex_predictions_tensor = torch.tensor(flex_predictions, dtype=torch.float32)
    flex_actual_values_tensor = torch.tensor(flex_actual_values, dtype=torch.float32)

    flex_difference = [RMSE(a, b) for a, b in zip(flex_predictions_tensor, flex_actual_values_tensor)]
    return (sum(flex_difference) / len(flex_difference)).item()
    

def get_mafe(data_arr, model, seq_len, error, boundary, time_horizon):
    flex_predictions = []
    flex_actual_values = []

    for data in data_arr:
        if len(data) < seq_len:
            continue

        data, _, _ = minmax_scale(data[:, 1:].astype(float))

        dataset = TimeSeriesDataset(data, seq_len, time_horizon, TARGET_COLUMN)
        dataloader = DataLoader(dataset, 1)

        for batch in dataloader:
            input_data, result_actual = batch  # Assuming the dataset returns a tuple (input_data, target_data)

            last_in_temp  = input_data[:, -1, 2:]

            lower_boundery = last_in_temp - boundary
            upper_boundery = last_in_temp + boundary

            result_predictions = model(input_data)

            actual_flex = flex_predict(result_actual[0], lower_boundery, upper_boundery, error)
            predicted_flex = flex_predict(result_predictions, lower_boundery, upper_boundery, error)

            flex_predictions.append(predicted_flex)
            flex_actual_values.append(actual_flex)
        
    flex_predictions_tensor = torch.tensor(flex_predictions, dtype=torch.float32)
    flex_actual_values_tensor = torch.tensor(flex_actual_values, dtype=torch.float32)

    flex_difference = [RMSE(a, b) for a, b in zip(flex_predictions_tensor, flex_actual_values_tensor)]
    return (sum(flex_difference) / len(flex_difference)).item()
    
    