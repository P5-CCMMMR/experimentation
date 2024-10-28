from cleaners.cleaner import Cleaner
from normalizers.normalizer import Normalizer
from splitters.splitter import Splitter
from model_handlers import Handler

# TODO 
# 1. Get the basic thing running
# 2. Use proper builder pattern
# 3. Make wrapper for tuner
# 4. Make train, test & val steps into modules
# 5. Make ensemble a handler instead of overall thing

class Pipeline:
    def __init__(self):
        self.df_arr = []
        self.model_arr = []

        self.train_df_arr = []
        self.val_df_arr   = []
        self.test_df_arr  = []
        self.norm_arr     = []

        self.cleaner = None
        self.splitter = None
        self.normalizer_class = None
        self.model = None
        self.trainer = None
        self.tester = None

    def __reset(self):
        self.model_arr = []

        self.train_df_arr = []
        self.val_df_arr   = []
        self.test_df_arr  = []
        self.norm_arr     = []

    def add_data(self, df):
        self.df_arr.append(df)
        return self

    def set_clean(self, cleaner):
        if not isinstance(cleaner, Cleaner):
            raise ValueError("Cleaner given not extended from Cleaner class")
        
        self.__reset()
        self.cleaner = cleaner
        return self
    
    def set_normalizer_class(self, normalizer_class):
        if not issubclass(normalizer_class, Normalizer):
            raise ValueError("Normalizer sub class given not extended from Normalizer class")
        
        self.__reset()
        self.normalizer_class = normalizer_class        
        return self
    
    def set_splitter(self, splitter):
        if not isinstance(splitter, Splitter):
            raise ValueError("Splitter given not extended from Splitter class")
        
        self.__reset()
        self.splitter = splitter
        return self
    
    # 1. model instance is deep copied | 2. model constructor is given and parameters are set seperately (going with 1 for now) 
    def set_model(self, model):
        if not isinstance(model, Handler):
            raise ValueError("Model Sub class given not extended from Model class")
        
        self.model_arr = []
        self.model = model 
        return self
    
    def set_tuner(self, tuner):
        if not isinstance(tuner, Tuner):
            raise ValueError("Trainer sub class given not extended from trainer class")
        
        self.model_arr = []
        self.tuner = tuner
        return self
    
    def run(self):
        for df in self.df_arr:
            df = self.cleaner.clean(df)

            # Problem with 1 normalizer is there is a connection between train, test og val
            normalizer = self.Normalizer(df) 
            df = normalizer.normalize()

            self.train_df_arr.append(self.splitter.get_train(df))
            self.val_df_arr.append(self.splitter.get_val(df))
            self.test_df_arr.append(self.splitter.get_test(df))
            self.norm_arr.append(normalizer)

        
        return self.tester.test(self.model_arr, self.test_df_arr)