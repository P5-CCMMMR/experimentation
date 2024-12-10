import numpy as np
import torch
from src.pipelines.baselines.deterministic_baseline import DeterministicBaseline

class LagDeterministicBaseline(DeterministicBaseline):  
    def forward(self, x):
        batch_arr = []
        for input in x:
            prediction_arr = []
            last_input_temp = input[len(input) - 1][self.target_column]
            for _ in range(0, self.horizen_len):
                prediction_arr.append(last_input_temp)
            batch_arr.append(prediction_arr)

        return torch.tensor(batch_arr)
    
    class Builder(DeterministicBaseline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = LagDeterministicBaseline

