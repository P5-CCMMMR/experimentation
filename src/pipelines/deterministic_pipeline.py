import copy
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

    def copy(self):
        new_instance = DeterministicPipeline(
            learning_rate=self.learning_rate,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            optimizer=copy.deepcopy(self.optimizer),
            model=copy.deepcopy(self.model),
            trainer=copy.deepcopy(self.trainer),
            tuner_class=self.tuner_class,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            test_timesteps=self.timesteps,
            normalizer=self.normalizer,
            train_error_func=self.train_error_func,
            val_error_func=self.val_error_func,
            test_error_func=self.test_error_func,
            target_column=self.target_column,
        )
        return new_instance
 
    class Builder(Pipeline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = DeterministicPipeline
    

