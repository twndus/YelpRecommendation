from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

class NSBCELoss(nn.BCELoss):

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, negative_mask: Tensor) -> Tensor:
        # make loss masking adding negative_mask to target and find nonzero indices
        loss_targets = (target.add(negative_mask)).nonzero(as_tuple=True)
        # compute loss only for nonzero indices
        return nn.functional.binary_cross_entropy(input[loss_targets], target[loss_targets], weight=self.weight, reduction=self.reduction)


class BPRLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, positive_preds, negative_preds):
        difference = positive_preds - negative_preds
        return torch.mean(-self.logsigmoid(difference))
