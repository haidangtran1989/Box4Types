import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TypeSelfAttentionLayer(nn.Module):
    def __init__(self,
                 scale: float = 1.0,
                 attn_dropout: float = 0.0):
        super(TypeSelfAttentionLayer, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = torch.matmul(q, k.transpose(1, 2)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn
