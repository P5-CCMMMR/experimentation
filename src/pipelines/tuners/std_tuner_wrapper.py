from .tuner_wrapper import TunerWrapper
from lightning.pytorch.tuner import Tuner

class StdTunerWrapper(TunerWrapper):

    def tune(self):
        tuner = Tuner(self.trainer)
        tuner.lr_find(self.model)
        tuner.scale_batch_size(self.model, mode="binsearch")
