class TrainerWrapper:
    def __init__(self, trainer_class, **kwargs):
        self.trainer = trainer_class(**kwargs)
        self.trainer_class = trainer_class
        self.kwargs = kwargs

    def copy(self, **new_kwargs):
        updated_kwargs = {**self.kwargs, **new_kwargs}
        return TrainerWrapper(self.trainer_class, **updated_kwargs)
    
    def get_trainer(self):
        return self.trainer