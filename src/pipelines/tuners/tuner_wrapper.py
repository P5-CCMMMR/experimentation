from abc import ABC, abstractmethod

class TunerWrapper(ABC):
    def __init__(self, trainer, model):
        self.trainer = trainer
        self.model = model

    @abstractmethod
    def tune(self):
        pass