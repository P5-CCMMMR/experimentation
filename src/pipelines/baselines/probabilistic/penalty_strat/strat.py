from abc import ABC, abstractmethod

class Strat(ABC):
    @staticmethod 
    @abstractmethod
    def calc():
        pass