import torch
from torch.nn import Module

from abc import ABC, abstractmethod

class BaseModel(Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _init_weights(self):
        pass
