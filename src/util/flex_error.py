import torch
from src.pipelines.sequencers.time_sequencer import TimeSequencer
from torch.utils.data import DataLoader
from src.pipelines.normalizers.min_max_normalizer import MinMaxNormalizer
from src.util.error import RMSE
from src.util.flex_predict import flex_predict, prob_flex_predict
from src.util.plot import plot_flex_probabilities

def get_prob_mafe(data_arr, model, seq_len, error, boundary, time_horizon, target_column, confidence=0.95):
    flex_actual_values = []
    flex_predictions = []
    flex_probabilities = []

    for data in data_arr:
        if len(data) < seq_len:
            continue

        normalizer = MinMaxNormalizer(data[:,1:].astype(float)) 

        data = normalizer.normalize()

        dataset = TimeSequencer(data[0], seq_len, time_horizon, target_column)
        dataloader = DataLoader(dataset, 1)

        for batch in dataloader:
            input_data, result_actual = batch  

            last_in_temp  = input_data[:, -1, 2:]

            lower_boundery = last_in_temp - boundary
            upper_boundery = last_in_temp + boundary

            result_predictions = model(input_data)

            actual_flex = flex_predict(result_actual[0], lower_boundery, upper_boundery, error)
            predicted_flex, probabilities = prob_flex_predict(result_predictions, lower_boundery, upper_boundery, error, confidence=confidence)
            
            flex_actual_values.append(actual_flex)
            flex_predictions.append(predicted_flex)
            flex_probabilities.append(probabilities)
        
    flex_predictions_tensor = torch.tensor(flex_predictions, dtype=torch.float32)
    flex_actual_values_tensor = torch.tensor(flex_actual_values, dtype=torch.float32)

    # Plot last flex probabilities
    plot_flex_probabilities(flex_probabilities[-1], confidence)

    flex_difference = [RMSE(a, b) for a, b in zip(flex_predictions_tensor, flex_actual_values_tensor)]
    return (sum(flex_difference) / len(flex_difference)).item()
    
def get_mafe(data_arr, model, seq_len, error, boundary, time_horizon, target_column):
    flex_predictions = []
    flex_actual_values = []

    for data in data_arr:
        if len(data) < seq_len:
            continue

        normalizer = MinMaxNormalizer(data[:,1:].astype(float)) 

        data = normalizer.normalize()

        dataset = TimeSequencer(data[0], seq_len, time_horizon, target_column)
        dataloader = DataLoader(dataset, 1)

        for batch in dataloader:
            input_data, result_actual = batch  # Assuming the dataset returns a tuple (input_data, target_data)

            last_in_temp = input_data[:, -1, 2:]

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
    