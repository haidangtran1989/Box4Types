import torch
import torch.nn as nn


class SimpleDecoder(nn.Module):
    def __init__(self, output_dim: int, answer_num: int):
        super(SimpleDecoder, self).__init__()
        self.answer_num = answer_num
        self.linear = nn.Linear(output_dim, answer_num, bias=False)

    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        output_embed = self.linear(inputs)
        return output_embed
