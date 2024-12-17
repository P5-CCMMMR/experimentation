from src.pipelines.pipeline import Pipeline

class DeterministicPipeline(Pipeline):
    def copy(self):
        new_model = self.model.copy()
        new_optimizer = self.optimizer.copy(new_model)
        new_trainer_wrapper = self.trainer_wrapper.copy()

        new_instance = DeterministicPipeline(
            learning_rate=self.learning_rate,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            optimizer=new_optimizer,
            model=new_model,
            trainer_wrapper=new_trainer_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            test_timesteps=self.timesteps,
            normalizer=self.normalizer,
            train_error_func=self.train_error_func,
            val_error_func=self.val_error_func,
            test_error_func_arr=self.test_error_func_arr,
            target_column=self.target_column,
            test_power=self.test_power,
            test_outdoor=self.test_outdoor,
            use_tuner=self.use_tuner
        )
        return new_instance
 
    class Builder(Pipeline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = DeterministicPipeline

        def set_error(self, error_func):
            assert error_func.is_deterministic(), "Error functions for Deterministic pipeline has to be deterministic"
            self.train_error_func = error_func
            self.val_error_func = error_func
            self.add_test_error(error_func)
            return self

        def set_train_error(self, error_func):
            assert error_func.is_deterministic(), "Error functions for Deterministic pipeline has to be deterministic"
            self.train_error_func = error_func
            return self

        def set_val_error(self, error_func):
            assert error_func.is_deterministic(), "Error functions for Deterministic pipeline has to be deterministic"
            self.val_error_func = error_func
            return self
        
        def add_test_error(self, error_func):
            assert error_func.is_deterministic(), "Error functions for Deterministic pipeline has to be deterministic"
            self.test_error_func_arr.append(error_func)
            return self
