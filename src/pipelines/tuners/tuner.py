from lightning.pytorch.tuner import Tuner as LTuner

class Tuner():
    def __init__(self, trainer, model):
        self.trainer = trainer
        self.model = model
    
    def tune(self):
        tuner = LTuner(self.trainer)
        lr_finder = tuner.lr_find(self.model)
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_finder_plot.png")
