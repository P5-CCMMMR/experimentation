import torch
from numpy import np

from .probabilistic_handler import ProbabilisticHandler

class MonteCarloHandler(ProbabilisticHandler):

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.train_error_func(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.val_error_func(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        x, y = batch
        mean_prediction, std_prediction = self.__predict_with_mc_dropout(x)
        negative_mean_log_likelihood = self.test_error_func(torch.tensor(mean_prediction, device=y.device), torch.tensor(std_prediction, device=y.device), y)
        self.log('mean_negative_log_likelihood', negative_mean_log_likelihood, on_step=True, logger=True, prog_bar=True)

        self.all_predictions[0].extend(mean_prediction.flatten())
        self.all_predictions[1].extend(std_prediction.flatten())
        self.all_actuals.extend(y.detach().cpu().numpy().flatten())
    
    def __predict_with_mc_dropout(self, x):
        self.model.train()
        predictions = []

        with torch.no_grad():
            for _ in range(self.test_sample_nbr):
                y_hat = self.model(x)
                predictions.append(y_hat.cpu().numpy())

        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)

        return mean_prediction, std_prediction
    
    def forward(self, batch):
        x = batch
        return self.__predict_with_mc_dropout(x)
    
    class Builder(ProbabilisticHandler.Builder):
        def __init__(self):
            super().__init__()
            self.inference_samples = None

        def set_inference_samples(self, inference_samples: int):
            self.inference_samples = inference_samples
            return self

        def build(self):
            self._check_none(
                handler=self.model,
                learning_rate=self.learning_rate,
                seq_len=self.seq_len,
                batch_size=self.batch_size,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                train_error_func=self.train_error_func,
                val_error_func=self.val_error_func,
                test_error_func=self.test_error_func,
                inference_samples=self.inference_samples
            )

            return MonteCarloHandler(
                self.handler,
                self.learning_rate,
                self.seq_len,
                self.batch_size,
                self.optimizer_list,
                self.train_loader,
                self.val_loader,
                self.test_loader,
                self.train_error_func,
                self.val_error_func,
                self.test_error_func,
                self.inference_samples
            )

