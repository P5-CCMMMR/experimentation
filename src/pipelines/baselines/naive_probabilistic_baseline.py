import numpy as np
import torch
from src.pipelines.baselines.probabilistic_baseline import ProbabilisticBaseline

class NaiveProbabilisticBaseline(ProbabilisticBaseline):  
#    def forward(self, x):
#        input = x[0]
#        actual_len = len(self.all_actuals)
#        T = (actual_len // self.horizen_len) - 1 # all actual * horizon_len (elements in array) / horizon_len (different starts) - K
#        mean_arr = []
#        std_dev_arr = []
#
#        start_step_index = T % self.horizen_len
#
#        last_input_temp = input[len(input) - 1][self.target_column]
#
#        residual_sum = 0
#
#        sigma = 0 if (T < 1) else np.sqrt(self.error_arr[start_step_index] / T) 
#        for i in range(0, self.horizen_len):
#            # update residual
#            before_index = self.horizen_len - i
#            current_input_temp = input[len(input) -  before_index][self.target_column]
#            current_actual_temp = self.all_actuals[actual_len - before_index]
#            residual = (current_actual_temp - current_input_temp) ** 2
#            residual_sum += residual
#
#            # make return arrays
#            std_dev = 0 if T < self.horizen_len else sigma * np.sqrt(i + 1)
#            mean_arr.append(last_input_temp)
#            std_dev_arr.append(std_dev)
#
#        self.error_arr[start_step_index] += residual_sum
#
#        return np.array(mean_arr), np.array(std_dev_arr)
    

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

        T = prediction_arr_length # amount of elements until now (is also amount of guesses until now)
        K = 1                     # amount of parameters
        M = 0                     # amount of missing data

        last_temp = input[len(input) - 1][self.target_column]

        sigma = np.sqrt(self.error_arr[self.step_start_index] / (T - M - K)) # should be 0 // -1 if no prior elements
        for i in range(0, self.horizen_len):
            std_dev = sigma * np.sqrt(i + 1)
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


