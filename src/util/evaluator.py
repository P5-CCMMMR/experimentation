import numpy as np
import multiprocessing

import torch
from src.pipelines.sequencers.all_time_sequencer import AllTimeSequencer
from torch.utils.data import DataLoader
from src.util import flex_predict
from src.util.flex_predict import prob_flex_predict, flex_predict
from src.util.plotly import plot_flex_probabilities

class Evaluator:
    def __init__(self, model, error, boundary):
        self.model = model
        self.error = error
        self.boundary = boundary

        self.flex_actual_values = []
        self.flex_probabilities = []
        self.flex_predictions = []

    def init_predictions(self, data, seq_len, time_horizon, target_column, confidence=0.95):
        if len(data) < (time_horizon + seq_len):
            print(f"data: {len(data)} < time_horizon: {time_horizon} + seq_len: {seq_len} \nNot enough data for prob_mafe")
            return 0

        self.flex_actual_values = []
        self.flex_probabilities = []
        self.flex_predictions = []
        
        print("DATA LEN: ", len(data))

        dataset = AllTimeSequencer(data, seq_len, time_horizon, target_column)
        dataloader = DataLoader(dataset, 1, num_workers=multiprocessing.cpu_count())

        for batch in dataloader:
            input_data, result_actual = batch  

            last_in_temp  = input_data[:, -1, target_column + 1:]

            lower_boundery = last_in_temp - self.boundary
            upper_boundery = last_in_temp + self.boundary

            result_predictions = self.model.forward(input_data)

            if isinstance(result_actual, torch.Tensor): result_actual = result_actual.squeeze()
            if isinstance(result_predictions, torch.Tensor): result_predictions = result_predictions.squeeze()

            if isinstance(result_predictions, tuple):
                actual_flex = flex_predict(result_actual, lower_boundery, upper_boundery, self.error)
                predicted_flex, probabilities = prob_flex_predict(result_predictions, lower_boundery, upper_boundery, self.error, confidence=confidence)

                self.flex_actual_values.append(actual_flex)
                self.flex_predictions.append(predicted_flex)
                self.flex_probabilities.append(probabilities)
            else:
                actual_flex = flex_predict(result_actual, lower_boundery, upper_boundery, self.error)
                predicted_flex = flex_predict(result_predictions, lower_boundery, upper_boundery, self.error)

                self.flex_predictions.append(predicted_flex)
                self.flex_actual_values.append(actual_flex)

        if isinstance(result_predictions, tuple):
            plot_flex_probabilities(self.flex_probabilities, confidence)

    def evaluate(self, error_func):
        if len(self.flex_predictions) != len(self.flex_actual_values):
             raise RuntimeError(f"predicted flex and actual flex was not the same length\n predicted_flex: {len(self.flex_predictions)} | actual_flex: {len(self.flex_actual_values)}")
        
        return error_func(np.array(self.flex_predictions), np.array(self.flex_actual_values))