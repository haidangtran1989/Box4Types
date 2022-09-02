import argparse
import torch
import torch.nn as nn
from transformers import AutoConfig
from typing import Dict, Optional, Tuple
from models.model_base import ModelBase, TRANSFORMER_MODELS
from modules.highway_network import HighwayNetwork
from modules.simple_decoder import SimpleDecoder


class TransformerVecModel(ModelBase):
    def __init__(self, args: argparse.Namespace, answer_num: int):
        super(TransformerVecModel, self).__init__()
        print("Initializing <{}> model...".format(args.model_type))
        _model_class, _tokenizer_class = TRANSFORMER_MODELS[args.model_type]
        self.transformer_tokenizer = _tokenizer_class.from_pretrained(
            args.model_type)
        self.transformer_config = AutoConfig.from_pretrained(args.model_type)
        self.encoder = _model_class.from_pretrained(args.model_type)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.avg_pooling = args.avg_pooling
        self.reduced_type_emb_dim = args.reduced_type_emb_dim
        self.n_negatives = args.n_negatives
        output_dim = self.transformer_config.hidden_size
        self.transformer_hidden_size = self.transformer_config.hidden_size
        self.encoder_layer_ids = args.encoder_layer_ids
        if self.encoder_layer_ids:
            self.layer_weights = nn.ParameterList(
                [nn.Parameter(torch.randn(1), requires_grad=True)
                 for _ in self.encoder_layer_ids])
        if args.reduced_type_emb_dim > 0:
            output_dim = args.reduced_type_emb_dim
            self.proj_layer = HighwayNetwork(self.transformer_hidden_size,
                                             output_dim,
                                             2,
                                             activation=nn.ReLU())
        self.activation = nn.ReLU()
        self.classifier = SimpleDecoder(output_dim, answer_num)

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
            targets: Optional[torch.Tensor] = None,
            batch_num: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            output_hidden_states=True if self.encoder_layer_ids else False)

        if self.avg_pooling:  # Averaging all hidden states
            outputs = (outputs[0] * inputs["attention_mask"].unsqueeze(-1)).sum(
                1) / inputs["attention_mask"].sum(1).unsqueeze(-1)
        else:  # Use [CLS]
            if self.encoder_layer_ids:
                _outputs = torch.zeros_like(outputs[0][:, 0, :])
                for i, layer_idx in enumerate(self.encoder_layer_ids):
                    _outputs += self.layer_weights[i] * outputs[2][layer_idx][:,
                                                        0, :]
                outputs = _outputs
            else:
                outputs = outputs[0][:, 0, :]

        outputs = self.dropout(outputs)

        if self.reduced_type_emb_dim > 0:
            outputs = self.proj_layer(outputs)

        logits = self.classifier(outputs)

        if targets is not None:
            if self.training and self.n_negatives > 0:
                pos_idx = torch.where(targets.sum(dim=0) > 0.)[0]
                neg_idx = torch.where(targets.sum(dim=0) == 0.)[0]
                if self.n_negatives < neg_idx.size()[0]:
                    neg_idx = neg_idx[
                        torch.randperm(len(neg_idx))[:self.n_negatives]]
                    sampled_idx = torch.cat([pos_idx, neg_idx], dim=0)
                    loss = self.define_loss(logits[:, sampled_idx],
                                            targets[:, sampled_idx])
                else:
                    loss = self.define_loss(logits, targets)
            else:
                loss = self.define_loss(logits, targets)
        else:
            loss = None
        return loss, logits

