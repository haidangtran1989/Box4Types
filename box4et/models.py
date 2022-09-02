import argparse
import os
import torch
import torch.nn as nn

from transformers import AutoConfig
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from typing import Dict, Optional, Tuple

# Import custom modules
from box_wrapper import BoxTensor
from box_wrapper import CenterSigmoidBoxTensor
from box_wrapper import CenterBoxTensor, ConstantBoxTensor
from modules import BCEWithLogProbLoss
from modules import BoxDecoder
from modules import HighwayNetwork
from modules import LinearProjection
from modules import SimpleFeedForwardLayer
from modules import SimpleDecoder
from constant import load_conditional_probs
from constant import load_marginals_probs
from constant import load_vocab_dict
from constant import TYPE_FILES
from constant import BASE_PATH

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


class TransformerBoxModel(TransformerVecModel):
    box_types = {
        "BoxTensor": BoxTensor,
        "CenterBoxTensor": CenterBoxTensor,
        "ConstantBoxTensor": ConstantBoxTensor,
        "CenterSigmoidBoxTensor": CenterSigmoidBoxTensor
    }

    def __init__(self, args: argparse.Namespace, answer_num: int):
        super(TransformerBoxModel, self).__init__(args, answer_num)
        self.goal = args.goal
        self.mc_box_type = args.mc_box_type
        self.type_box_type = args.type_box_type
        self.box_offset = args.box_offset
        self.alpha_type_reg = args.alpha_type_reg
        self.alpha_type_vol_l1 = args.alpha_type_vol_l1
        self.alpha_type_vol_l2 = args.alpha_type_vol_l2
        self.th_type_vol = args.th_type_vol
        self.inv_softplus_temp = args.inv_softplus_temp
        self.softplus_scale = args.softplus_scale
        self.alpha_hierarchy_loss = args.alpha_hierarchy_loss

        try:
            self.mc_box = self.box_types[args.mc_box_type]
        except KeyError as ke:
            raise ValueError(
                "Invalid box type {}".format(args.box_type)) from ke

        if args.proj_layer == "linear":
            self.proj_layer = LinearProjection(
                self.transformer_hidden_size,
                args.box_dim * 2)
        elif args.proj_layer == "mlp":
            self.proj_layer = SimpleFeedForwardLayer(
                self.transformer_hidden_size,
                args.box_dim * 2,
                activation=nn.Sigmoid())
        elif args.proj_layer == "highway":
            self.proj_layer = HighwayNetwork(
                self.transformer_hidden_size,
                args.box_dim * 2,
                args.n_proj_layer,
                activation=nn.ReLU())
        else:
            raise ValueError(args.proj_layer)

        if args.pretrained_box_path:
            print("Loading pretrained box emb from {}".format(
                args.pretrained_box_path))
            pretrained_box_tsr = torch.load(args.pretrained_box_path).to(
                args.device)
        else:
            print("Not loading pretrained box emb.")
            pretrained_box_tsr = None
        self.classifier = BoxDecoder(answer_num,
                                     args.box_dim,
                                     args.type_box_type,
                                     inv_softplus_temp=args.inv_softplus_temp,
                                     softplus_scale=args.softplus_scale,
                                     n_negatives=args.n_negatives,
                                     neg_temp=args.neg_temp,
                                     box_offset=args.box_offset,
                                     pretrained_box=pretrained_box_tsr,
                                     use_gumbel_baysian=args.use_gumbel_baysian,
                                     gumbel_beta=args.gumbel_beta)
        self.loss_func = BCEWithLogProbLoss()

        if args.marginal_prob_path:
            self.type_marginals = load_marginals_probs(
                os.path.join(BASE_PATH, args.marginal_prob_path),
                args.device)

        if args.conditional_prob_path:
            word2id = load_vocab_dict(TYPE_FILES[args.goal])
            self.type_pairs, self.type_pairs_conditional = load_conditional_probs(
                os.path.join(BASE_PATH, args.conditional_prob_path),
                word2id,
                args.device)
            print("Loaded type pairs:",
                  self.type_pairs.size(),
                  self.type_pairs_conditional.size())
            self.hierarchy_loss_func = torch.nn.KLDivLoss(reduction="batchmean")

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
            targets: Optional[torch.Tensor] = None,
            batch_num: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mention_context_rep = self.encoder(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            output_hidden_states=True if self.encoder_layer_ids else False)

        # CLS
        if self.encoder_layer_ids:
            # Weighted sum of CLS from different layers
            _mention_context_rep = torch.zeros_like(
                mention_context_rep[0][:, 0, :])
            for i, layer_idx in enumerate(self.encoder_layer_ids):
                _mention_context_rep += self.layer_weights[i] * \
                                        mention_context_rep[2][layer_idx][:, 0, :]
            mention_context_rep = _mention_context_rep
        else:
            # CLS from the last layer
            mention_context_rep = mention_context_rep[0][:, 0, :]

        # Convert to box
        mention_context_rep = self.proj_layer(mention_context_rep)
        if self.mc_box_type == 'ConstantBoxTensor':

            mention_context_rep = self.mc_box.from_split(mention_context_rep,
                                                         self.box_offset)
        else:
            mention_context_rep = self.mc_box.from_split(mention_context_rep)
        # Compute probs (0-1 scale)
        log_probs, loss_weights, _ = self.classifier(mention_context_rep,
                                                     targets=targets,
                                                     is_training=self.training,
                                                     batch_num=batch_num)
        return None, torch.exp(log_probs)
