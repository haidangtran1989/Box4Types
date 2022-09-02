import torch
import torch.nn as nn


class LinearProjection(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 bias: bool = True):
        super(LinearProjection, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(inputs)
        return outputs
