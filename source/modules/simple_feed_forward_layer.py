import torch
import torch.nn as nn
from typing import Optional


class SimpleFeedForwardLayer(nn.Module):
    """2-layer feed forward"""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 bias: bool = True,
                 activation: Optional[nn.Module] = None):
        super(SimpleFeedForwardLayer, self).__init__()
        self.linear_projection1 = nn.Linear(input_dim,
                                            (input_dim + output_dim) // 2,
                                            bias=bias)
        self.linear_projection2 = nn.Linear((input_dim + output_dim) // 2,
                                            output_dim,
                                            bias=bias)
        self.activation = activation if activation else nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.activation(self.linear_projection1(inputs))
        inputs = self.activation(self.linear_projection2(inputs))
        return inputs
