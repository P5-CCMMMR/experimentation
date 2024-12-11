import numpy as np
import torch
from src.pipelines.baselines.deterministic.deterministic_baseline import DeterministicBaseline

class LagDeterministicBaseline(DeterministicBaseline):  
    def forward(self, x):
        prediction_arr = []
        for input in x:
            last_input_temp = input[len(input) - 1][self.target_column]
            for _ in range(0, self.horizen_len):
                prediction_arr.append(last_input_temp)
        return prediction_arr
    
    class Builder(DeterministicBaseline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = LagDeterministicBaseline

