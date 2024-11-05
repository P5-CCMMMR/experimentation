class OptimizerWrapper:
    def __init__(self, optimizer_class, model, **kwargs):
        self.optimizer = optimizer_class(model.parameters(), **kwargs)
        self.optimizer_class = optimizer_class
        self.kwargs = kwargs

    def copy(self, new_model):
        return OptimizerWrapper(self.optimizer_class, new_model, **self.kwargs)
    
    def get_optimizer(self):
        return self.optimizer
