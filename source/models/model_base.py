import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from typing import Dict, Optional

TRANSFORMER_MODELS = {
    "bert-base-uncased": (BertModel, BertTokenizer),
    "bert-large-uncased": (BertModel, BertTokenizer),
    "bert-large-uncased-whole-word-masking": (BertModel, BertTokenizer),
    "roberta-base": (RobertaModel, RobertaTokenizer),
    "roberta-large": (RobertaModel, RobertaTokenizer)
}


class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss()
        self.sigmoid_fn = nn.Sigmoid()

    def define_loss(self,
                    logits: torch.Tensor,
                    targets: torch.Tensor,
                    weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is not None:
            loss = self.loss_func(logits, targets, weight=weight)
        else:
            loss = self.loss_func(logits, targets)
        return loss

    def forward(self, feed_dict: Dict[str, torch.Tensor]):
        pass


