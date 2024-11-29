from abc import ABC, abstractmethod

class Cleaner(ABC):
    
    @abstractmethod
    def clean(self, df):
        pass