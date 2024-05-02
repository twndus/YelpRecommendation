# Base module for Dataset class

import torch
from torch.utils.data import Dataset
from abc import abstractmethod

class DataPipeline:
    '''
        split
            random user
            temporal user
            temporal global
        preprocess
            Dataset으로 만드는 거 -> task별로 달라지지 abstractmethod로 만들어두기
        
    '''
    @abstractmethod
    def split(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass