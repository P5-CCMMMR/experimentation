from src.pipelines.splitters.splitter import Splitter
import pandas as pd

class BlockedKFoldSplitter(Splitter):
    def __init__(self, folds):
        super().__init__(0, 0, 0)
        self.folds = folds
        self.val_index = 0

    def set_val_index(self, index):
        assert index < self.folds, "Index larger than folds can not be given"
        self.val_index = index

    def get_train(self, df: pd.DataFrame) -> pd.DataFrame:
        fold_size = len(df) // self.folds
        start = self.val_index * fold_size
        end = (self.val_index + 1) * fold_size
        df = df.drop(df.index[start:end])
        return df
    
    def get_val(self, df: pd.DataFrame) -> pd.DataFrame:
        fold_size = len(df) // self.folds
        start = self.val_index * fold_size
        end = (self.val_index + 1) * fold_size
        return df.iloc[start:end]
    
    def get_test(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()
    
