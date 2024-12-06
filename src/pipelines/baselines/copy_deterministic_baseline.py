from src.pipelines.baselines.deterministic_baseline import DeterministicBaseline

class CopyDeterministicBaseline(DeterministicBaseline):
    def forward(self, x):
        prediction_arr = []
        for input in x:
            for i in range(0, self.horizen_len):
                before_index = self.horizen_len - i 
                prediction_arr.append(input[len(input) - before_index][self.target_column])
                
        return prediction_arr
 
    class Builder(DeterministicBaseline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = CopyDeterministicBaseline
