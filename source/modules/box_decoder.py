import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from utils.box_wrapper import BoxTensor, log1mexp
from utils.box_wrapper import CenterBoxTensor
from utils.box_wrapper import ConstantBoxTensor, CenterSigmoidBoxTensor
from modules.type_self_attention_layer import TypeSelfAttentionLayer
from utils.mention_context_similarity import get_mention_context_similarity

euler_gamma = 0.57721566490153286060


def _compute_gumbel_min_max(
        box1: BoxTensor,
        box2: BoxTensor,
        gumbel_beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns min and max points."""
    min_point = torch.stack([box1.z, box2.z])
    min_point = torch.max(
        gumbel_beta * torch.logsumexp(min_point / gumbel_beta, 0),
        torch.max(min_point, 0)[0])

    max_point = torch.stack([box1.Z, box2.Z])
    max_point = torch.min(
        -gumbel_beta * torch.logsumexp(-max_point / gumbel_beta, 0),
        torch.min(max_point, 0)[0])
    return min_point, max_point


def _compute_hard_min_max(
        box1: BoxTensor,
        box2: BoxTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns min and max points."""
    min_point = torch.max(box1.z, box2.z)
    max_point = torch.min(box1.Z, box2.Z)
    return min_point, max_point


class BoxDecoder(nn.Module):
    box_types = {
        'BoxTensor': BoxTensor,
        'CenterBoxTensor': CenterBoxTensor,
        'ConstantBoxTensor': ConstantBoxTensor,
        'CenterSigmoidBoxTensor': CenterSigmoidBoxTensor
    }

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 box_type: str,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 _weight: Optional[torch.Tensor] = None,
                 init_interval_delta: float = 0.5,
                 init_interval_center: float = 0.01,
                 inv_softplus_temp: float = 1.,
                 softplus_scale: float = 1.,
                 n_negatives: int = 0,
                 neg_temp: float = 0.,
                 box_offset: float = 0.5,
                 pretrained_box: Optional[torch.Tensor] = None,
                 use_gumbel_baysian: bool = False,
                 gumbel_beta: float = 1.0):
        super(BoxDecoder, self).__init__()

        self.num_embeddings = num_embeddings
        self.box_embedding_dim = embedding_dim
        self.box_type = box_type
        try:
            self.box = self.box_types[box_type]
        except KeyError as ke:
            raise ValueError("Invalid box type {}".format(box_type)) from ke
        self.box_offset = box_offset  # Used for constant tensor
        self.init_interval_delta = init_interval_delta
        self.init_interval_center = init_interval_center
        self.inv_softplus_temp = inv_softplus_temp
        self.softplus_scale = softplus_scale
        self.n_negatives = n_negatives
        self.neg_temp = neg_temp
        self.use_gumbel_baysian = use_gumbel_baysian
        self.gumbel_beta = gumbel_beta
        self.box_embeddings = nn.Embedding(num_embeddings,
                                           embedding_dim * 2,
                                           padding_idx=padding_idx,
                                           max_norm=max_norm,
                                           norm_type=norm_type,
                                           scale_grad_by_freq=scale_grad_by_freq,
                                           sparse=sparse,
                                           _weight=_weight)
        self.type_attention = TypeSelfAttentionLayer()  # not in use
        if pretrained_box is not None:
            print('Init box emb with pretrained boxes.')
            print(self.box_embeddings.weight)
            self.box_embeddings.weight = nn.Parameter(pretrained_box)
            print(self.box_embeddings.weight)

    def init_weights(self):
        print('before', self.box_embeddings.weight)
        torch.nn.init.uniform_(
            self.box_embeddings.weight[..., :self.box_embedding_dim],
            -self.init_interval_center, self.init_interval_center)
        torch.nn.init.uniform_(
            self.box_embeddings.weight[..., self.box_embedding_dim:],
            self.init_interval_delta, self.init_interval_delta)
        print('after', self.box_embeddings.weight)

    def log_soft_volume(
            self,
            z: torch.Tensor,
            Z: torch.Tensor,
            temp: float = 1.,
            scale: float = 1.,
            gumbel_beta: float = 0.) -> torch.Tensor:
        eps = torch.finfo(z.dtype).tiny  # type: ignore

        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale

        if gumbel_beta <= 0.:
            return (torch.sum(
                torch.log(F.softplus(Z - z, beta=temp).clamp_min(eps)),
                dim=-1) + torch.log(s)
                    )  # need this eps to that the derivative of log does not blow
        else:
            return (torch.sum(
                torch.log(
                    F.softplus(Z - z - 2 * euler_gamma * gumbel_beta, beta=temp).clamp_min(
                        eps)),
                dim=-1) + torch.log(s))

    def type_box_volume(self) -> torch.Tensor:
        inputs = torch.arange(0,
                              self.box_embeddings.num_embeddings,
                              dtype=torch.int64,
                              device=self.box_embeddings.weight.device)
        emb = self.box_embeddings(inputs)  # num types x 2*box_embedding_dim
        if self.box_type == 'ConstantBoxTensor':
            type_box = self.box.from_split(emb, self.box_offset)
        else:
            type_box = self.box.from_split(emb)

        vol = self.log_soft_volume(type_box.z,
                                   type_box.Z,
                                   temp=self.inv_softplus_temp,
                                   scale=self.softplus_scale,
                                   gumbel_beta=self.gumbel_beta)
        return vol

    def get_pairwise_conditional_prob(self,
                                      type_x_ids: torch.Tensor,
                                      type_y_ids: torch.Tensor) -> torch.Tensor:
        inputs = torch.arange(0,
                              self.box_embeddings.num_embeddings,
                              dtype=torch.int64,
                              device=self.box_embeddings.weight.device)
        emb = self.box_embeddings(inputs)  # num types x 2*box_embedding_dim
        type_x = emb[type_x_ids]
        type_y = emb[type_y_ids]
        type_x_box = self.box.from_split(type_x)
        type_y_box = self.box.from_split(type_y)

        # Compute intersection volume
        if self.use_gumbel_baysian:
            # Gumbel intersection
            min_point, max_point = _compute_gumbel_min_max(type_x_box,
                                                           type_y_box,
                                                           self.gumbel_beta)
        else:
            min_point, max_point = _compute_hard_min_max(type_x_box, type_y_box)

        intersection_vol = self.log_soft_volume(min_point,
                                                max_point,
                                                temp=self.inv_softplus_temp,
                                                scale=self.softplus_scale,
                                                gumbel_beta=self.gumbel_beta)
        # Compute y volume here
        y_vol = self.log_soft_volume(type_y_box.z,
                                     type_y_box.Z,
                                     temp=self.inv_softplus_temp,
                                     scale=self.softplus_scale,
                                     gumbel_beta=self.gumbel_beta)

        # Need to be careful about numerical issues
        conditional_prob = intersection_vol - y_vol
        return torch.cat([conditional_prob.unsqueeze(-1),
                          log1mexp(conditional_prob).unsqueeze(-1)],
                         dim=-1)

    def forward(
            self,
            mention_context_box: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            is_training: bool = True,
            batch_num: Optional[int] = None
    ) -> Tuple[torch.Tensor, None]:
        inputs = torch.arange(0,
                              self.box_embeddings.num_embeddings,
                              dtype=torch.int64,
                              device=self.box_embeddings.weight.device)
        emb = self.box_embeddings(inputs)  # num types x 2*box_embedding_dim
        type_box = self.box.from_split(emb)
        return get_mention_context_similarity(mention_context_box, type_box)


def get_classifier(args, answer_num):
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
