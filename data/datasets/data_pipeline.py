# Base module for Dataset class
from abc import ABC, abstractmethod

class DataPipeline(ABC):
    '''
        split
            random user
            temporal user
            temporal global
        preprocess
            Dataset으로 만드는 거 -> task별로 달라지지 abstractmethod로 만들어두기
    '''

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def split(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass
