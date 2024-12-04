import numpy as np
import torch
from src.pipelines.deterministic_baseline import DeterministicBaseline

class LagDeterministicBaseline(DeterministicBaseline):
    def __init__(self, target_column, test_loader, test_timestamps, test_normalizer, test_error_func_arr, horizon_len):
        super().__init__(target_column, test_loader, test_timestamps, test_normalizer, test_error_func_arr, horizon_len)
        
    def test(self):

        for batch in self.test_loader:
            x, y = batch
            self.all_predictions.append(self.forward(x.detach().cpu().numpy()))
            self.all_actuals.append(y.detach().cpu().numpy().flatten())

        self.all_predictions = self.normalizer.denormalize(np.array(self.all_predictions), self.target_column)
        self.all_actuals = self.normalizer.denormalize(np.array(self.all_actuals), self.target_column)

        func_arr = self.test_error_func_arr
        for func in func_arr:
            loss = func.calc(torch.tensor(self.all_predictions), torch.tensor(self.all_actuals))
            title = func.get_title()
            print(f"{title:<30} {loss:.6f}")

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

        def build(self):    
            _, _, test_loader, test_timestamps, test_normalizer = self._get_loaders(self.horizon_len)
            return self.pipeline_class(self.target_column, test_loader, test_timestamps, test_normalizer, self.test_error_func_arr, self.horizon_len)
        
    


