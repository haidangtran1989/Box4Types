import torch
import torch.nn.functional as functional


INV_SOFTPLUS_TEMP = 1.2471085395024732
SOFTPLUS_SCALE = 1.0
GUMBEL_BETA = 0.00036026463511690845
EULER_GAMMA = 0.57721566490153286060


def get_mention_context_similarity(mention_context_box1, mention_context_box2):
    size1 = mention_context_box1.data.size()[0]
    size2 = mention_context_box2.data.size()[0]
    min_point = torch.max(torch.stack([mention_context_box1.z.unsqueeze(1).expand(-1, size2, -1),
                                       mention_context_box2.z.unsqueeze(0).expand(size1, -1, -1)]), 0)[0]
    max_point = torch.min(torch.stack([mention_context_box1.Z.unsqueeze(1).expand(-1, size2, -1),
                                       mention_context_box2.Z.unsqueeze(0).expand(size1, -1, -1)]), 0)[0]
    # Get soft volume
    # Compute the volume of the intersection
    intersection_volume = log_soft_volume(min_point, max_point)
    # Compute  the volume of the first mention context box
    volume1 = log_soft_volume(mention_context_box1.z, mention_context_box1.Z)
    # Returns log probability list
    log_probs = intersection_volume - volume1.unsqueeze(-1)
    # Clip values > 1 for numerical stability
    if (log_probs > 0.0).any():
        print("WARNING: Clipping log probability since it's grater than 0")
        log_probs[log_probs > 0.0] = 0.0
    return log_probs, None, None


def log_soft_volume(point1: torch.Tensor, point2: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(point1.dtype).tiny
    if isinstance(SOFTPLUS_SCALE, float):
        softplus_scale = torch.tensor(SOFTPLUS_SCALE)
    else:
        softplus_scale = SOFTPLUS_SCALE
    if GUMBEL_BETA <= 0.:
        return torch.sum(torch.log(functional.softplus(point2 - point1, beta=INV_SOFTPLUS_TEMP).clamp_min(eps)), dim=-1) + torch.log(softplus_scale)
    else:
        return torch.sum(torch.log(functional.softplus(point2 - point1 - 2 * EULER_GAMMA * GUMBEL_BETA, beta=INV_SOFTPLUS_TEMP).clamp_min(eps)), dim=-1) + torch.log(softplus_scale)
