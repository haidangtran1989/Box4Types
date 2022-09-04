import argparse
from typing import Dict, Optional, Tuple
import torch
from models.transformer_box_model import TransformerBoxModel
from modules.box_decoder import BoxDecoder


def create_classifier(args, answer_num):
    return BoxDecoder(answer_num,
                                 args.box_dim,
                                 args.type_box_type,
                                 inv_softplus_temp=args.inv_softplus_temp,
                                 softplus_scale=args.softplus_scale,
                                 n_negatives=args.n_negatives,
                                 neg_temp=args.neg_temp,
                                 box_offset=args.box_offset,
                                 pretrained_box=None,
                                 use_gumbel_baysian=args.use_gumbel_baysian,
                                 gumbel_beta=args.gumbel_beta)


class TransformerBoxModelV1(TransformerBoxModel):
    def __init__(self, args: argparse.Namespace, answer_num: int):
        super(TransformerBoxModelV1, self).__init__()
        self.classifier = create_classifier(args, answer_num)

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
            targets: Optional[torch.Tensor] = None,
            batch_num: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mention_context_rep = self.build_representation(inputs)
        log_probs, loss_weights, _ = self.classifier(mention_context_rep,
                                                     targets=targets,
                                                     is_training=self.training,
                                                     batch_num=batch_num)
        # x, _, _ = get_mention_context_similarity(mention_context_rep, mention_context_rep)
        # x = torch.exp(x)
        return torch.exp(log_probs)
