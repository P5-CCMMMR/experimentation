import numpy as np
import torch
from src.pipelines.baselines.probabilistic.probabilistic_baseline import ProbabilisticBaseline

class NaiveProbabilisticBaseline(ProbabilisticBaseline):  
    def forward(self, x):
        input = x[0]
        prediction_arr = self.forward_prediction_2d_arr[self.step_start_index]
        prediction_arr_length = len(prediction_arr)
        
        # update resduals
        residual_sum = 0
        if prediction_arr_length > 0:
            for i in range(0, self.horizen_len):
                before_index = self.horizen_len - i
                current_prediction_temp = prediction_arr[prediction_arr_length -  before_index]
                current_actual_temp = input[len(input) -  before_index][self.target_column]
                residual = (current_actual_temp - current_prediction_temp) ** 2
                residual_sum += residual

            self.error_arr[self.step_start_index] += residual_sum
 
        # make return arrays
        mean_arr = []
        std_dev_arr = []

        T = prediction_arr_length 
  
        last_temp = input[len(input) - 1][self.target_column]
        
        sigma = 0 if T == 0 else np.sqrt(self.error_arr[self.step_start_index] / T) 
        for i in range(0, self.horizen_len):
            std_dev = sigma * self.penalty_strat.calc(i + 1, T)
            mean_arr.append(last_temp)
            std_dev_arr.append(std_dev)

        # Prepare for next forward and next forward in this step
        self.forward_prediction_2d_arr[self.step_start_index].extend(mean_arr)
        self.step_start_index = (self.step_start_index + 1) % self.horizen_len

        return np.array(mean_arr), np.array(std_dev_arr)

    class Builder(ProbabilisticBaseline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = NaiveProbabilisticBaseline


