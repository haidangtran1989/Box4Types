import argparse
import os
from typing import Dict, Optional, Tuple
import torch
from torch import nn as nn
from box_wrapper import BoxTensor, CenterBoxTensor, ConstantBoxTensor, CenterSigmoidBoxTensor
from constant import load_marginals_probs, BASE_PATH, load_vocab_dict, TYPE_FILES, load_conditional_probs
from models.transformer_vec_model import TransformerVecModel
from modules.bce_with_log_prob_loss import BCEWithLogProbLoss
from modules.box_decoder import BoxDecoder
from modules.highway_network import HighwayNetwork
from modules.linear_projection import LinearProjection
from modules.simple_feed_forward_layer import SimpleFeedForwardLayer
from utils.mention_context_similarity import get_mention_context_similarity


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
        mention_context_rep = self.get_box_representation(inputs)
        log_probs, loss_weights, _ = self.classifier(mention_context_rep,
                                                     targets=targets,
                                                     is_training=self.training,
                                                     batch_num=batch_num)
        # x, _, _ = get_mention_context_similarity(mention_context_rep, mention_context_rep)
        # x = torch.exp(x)
        return None, torch.exp(log_probs)

    def get_box_representation(self, inputs):
        mention_context_rep = self.encoder(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            output_hidden_states=True if self.encoder_layer_ids else False)
        mention_context_rep = mention_context_rep[0][:, 0, :]
        # Convert CLS embedding to box
        mention_context_rep = self.proj_layer(mention_context_rep)
        mention_context_rep = self.mc_box.from_split(mention_context_rep)
        # Compute probability list (0-1 scale)
        return mention_context_rep
