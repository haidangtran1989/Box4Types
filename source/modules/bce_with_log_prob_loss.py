import torch
import torch.nn as nn
from typing import Optional
from box_wrapper import log1mexp


class BCEWithLogProbLoss(nn.BCELoss):
    def _binary_cross_entropy(self,
                              input: torch.Tensor,
                              target: torch.Tensor,
                              weight: Optional[torch.Tensor] = None,
                              reduction: str = 'mean') -> torch.Tensor:
        """Computes binary cross entropy.

        This function takes log probability and computes binary cross entropy.

        Args:
          input: Torch float tensor. Log probability. Same shape as `target`.
          target: Torch float tensor. Binary labels. Same shape as `input`.
          weight: Torch float tensor. Scaling loss if this is specified.
          reduction: Reduction method. 'mean' by default.
        """
        loss = -target * input - (1 - target) * log1mexp(input)

        if weight is not None:
            loss = loss * weight

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, input, target, weight=None):
        return self._binary_cross_entropy(input,
                                          target,
                                          weight=weight,
                                          reduction=self.reduction)
