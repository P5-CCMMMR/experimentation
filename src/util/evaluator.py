import math
import numpy as np
import pandas as pd
import torch
from src.pipelines.sequencers.time_sequencer import TimeSequencer
from torch.utils.data import DataLoader
from src.util import flex_predict
from src.util.flex_predict import prob_flex_predict, flex_predict
from src.util.plotly import plot_flex_probabilities

class Evaluator:
    def __init__(self, model, error, boundary, error_func):
        self.model = model
        self.error = error
        self.boundary = boundary
        self.error_func = error_func

    def evaluate(self, data, seq_len, time_horizon, target_column, confidence=None):
        if len(data) < (time_horizon + seq_len):
            print(f"data: {len(data)} < time_horizon: {time_horizon} + seq_len: {seq_len} \nNot enough data for prob_mafe")
            return 0

        flex_actual_values = []
        flex_probabilities = []
        flex_predictions = []

        dataset = TimeSequencer(data, seq_len, time_horizon, target_column)
        dataloader = DataLoader(dataset, 1)

        for batch in dataloader:
            input_data, result_actual = batch  

            last_in_temp  = input_data[:, -1, target_column + 1:]

            lower_boundery = last_in_temp - self.boundary
            upper_boundery = last_in_temp + self.boundary
            result_predictions = self.model(input_data)

            if (confidence != None):
                actual_flex = flex_predict(result_actual[0], lower_boundery, upper_boundery, self.error)
                predicted_flex, probabilities = prob_flex_predict(result_predictions, lower_boundery, upper_boundery, self.error, confidence=confidence)

                flex_actual_values.append(actual_flex)
                flex_predictions.append(predicted_flex)
                flex_probabilities.append(probabilities)
            else:
                actual_flex = flex_predict(result_actual[0], lower_boundery, upper_boundery, self.error)
                predicted_flex = flex_predict(result_predictions, lower_boundery, upper_boundery, self.error)

                flex_predictions.append(predicted_flex)
                flex_actual_values.append(actual_flex)
        
        flex_predictions_array = np.array(flex_predictions, dtype=np.float32)
        flex_actual_values_array = np.array(flex_actual_values, dtype=np.float32)

        if len(flex_predictions_array) is not len(flex_actual_values_array):
             raise RuntimeError("predicted flex and actual flex was not the same length ")
        flex_difference = []
        
        for i in range(0, len(flex_actual_values_array)):
            flex_difference.append(self.error_func(flex_predictions_array[i], flex_actual_values_array[i]))

        return (sum(flex_difference) / len(flex_difference))