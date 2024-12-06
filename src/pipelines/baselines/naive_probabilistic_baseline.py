import numpy as np
import torch
from src.pipelines.baselines.probabilistic_baseline import ProbabilisticBaseline

class NaiveProbabilisticBaseline(ProbabilisticBaseline):  
    def forward(self, x):
        input = x[0]
        actual_len = len(self.all_actuals)
        T = (actual_len // self.horizen_len) - 1 # all actual * horizon_len (elements in array) / horizon_len (different starts) - K
        mean_arr = []
        std_dev_arr = []

        start_step_index = T % self.horizen_len

        last_input_temp = input[len(input) - 1][self.target_column]

        residual_sum = 0

        sigma = 0 if (T < 1) else np.sqrt(self.error_arr[start_step_index] / T) 
        for i in range(0, self.horizen_len):
            # update residual
            before_index = self.horizen_len - i
            current_input_temp = input[len(input) -  before_index][self.target_column]
            current_actual_temp = self.all_actuals[actual_len - before_index]
            residual = (current_actual_temp - current_input_temp) ** 2
            residual_sum += residual

            # make return arrays
            std_dev = 0 if T < self.horizen_len else sigma * np.sqrt(i + 1)
            mean_arr.append(last_input_temp)
            std_dev_arr.append(std_dev)

        self.error_arr[start_step_index] += residual_sum

        return np.array(mean_arr), np.array(std_dev_arr)
    
    class Builder(ProbabilisticBaseline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = NaiveProbabilisticBaseline

