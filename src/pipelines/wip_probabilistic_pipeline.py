import torch
from src.pipelines.wip_pipeline import Pipeline

class ProbabilisticPipeline(Pipeline):
    def training_step(self, batch):
        x, y = batch
        mean_prediction, std_prediction = self.forward(x)
        loss = self.test_error_func(torch.tensor(mean_prediction, device=y.device), torch.tensor(std_prediction, device=y.device), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        mean_prediction, std_prediction = self.forward(x)
        loss = self.test_error_func(torch.tensor(mean_prediction, device=y.device), torch.tensor(std_prediction, device=y.device), y)
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        x, y = batch
        mean_prediction, std_prediction = self.forward(x)
        loss = self.test_error_func(torch.tensor(mean_prediction, device=y.device), torch.tensor(std_prediction, device=y.device), y)
        self.log('test_loss', loss, on_step=True, logger=True, prog_bar=True)
        
        self.all_predictions[0].extend(mean_prediction.flatten())
        self.all_predictions[1].extend(std_prediction.flatten())
        self.all_actuals.extend(y.detach().cpu().numpy().flatten())
    
    class Builder(Pipeline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = ProbabilisticPipeline
    
