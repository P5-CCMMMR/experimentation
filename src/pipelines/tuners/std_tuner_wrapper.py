from .tuner_wrapper import TunerWrapper
from lightning.pytorch.tuner import Tuner

class StdTunerWrapper(TunerWrapper):

    def tune(self, scale_batch_size=False):
        tuner = Tuner(self.trainer)
        if scale_batch_size:
            tuner.scale_batch_size(self.model)
        tuner.lr_find(self.model)
