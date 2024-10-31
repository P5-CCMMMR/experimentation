from src.pipelines.pipeline import Pipeline

class DeterministicPipeline(Pipeline):
    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.train_error_func(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.val_error_func(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.test_error_func(y_hat, y)
        self.log('test_loss', loss, on_step=True, logger=True, prog_bar=True)
        
        self.all_predictions.extend(y_hat.detach().cpu().numpy().flatten())
        self.all_actuals.extend(y.detach().cpu().numpy().flatten())
 
    class Builder(Pipeline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = DeterministicPipeline
    

