from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class ConditionalEarlyStopping(EarlyStopping):
    """
    Enabling early stopping when the monitored metric is below a certain threshold.
    """
    def __init__(self, threshold: float, monitor='val_loss', min_delta=0.0, patience=3, verbose=False, mode='min', strict=True):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, strict=strict)
        self.threshold = threshold
        self.enabled = False

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current = logs.get(self.monitor)

        if current is None:
            if self.strict:
                raise RuntimeError(f'Metric `{self.monitor}` not found in logs.')
            return

        if current < self.threshold:
            self.enabled = True

        if current >= self.threshold:
            self.enabled = False
            self.wait_count = 0
            
        if self.enabled:
            super().on_validation_end(trainer, pl_module)