import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _activation_module(function_name:str) -> nn.Module:
        if function_name == 'sigmoid':
            return nn.Sigmoid()
        elif function_name == 'identity':
            return nn.Identity()

    @abstractmethod
    def _init_weights(self):
        pass
