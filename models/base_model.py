import torch
from torch.nn import Module

from abc import abstractmethod

class BaseModel(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _init_weights(self):
        pass