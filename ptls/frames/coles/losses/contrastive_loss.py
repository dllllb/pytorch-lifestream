import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.distributed as dist

class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return tuple(gathered)

    @staticmethod
    def backward(ctx, *grad_outs):
        # if os.environ.get('REDUCE_GRADS'):
        # grad_outs = torch.stack(grad_outs)
        # dist.all_reduce(grad_outs)
        return grad_outs[dist.get_rank()]

def all_gather_and_cat(tensor):
    return torch.cat(AllGather.apply(tensor))


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss

    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(self, margin, sampling_strategy, distributed_mode = False, use_gpu_dependent_labels = False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = sampling_strategy
        self.use_gpu_dependent_labels = use_gpu_dependent_labels
        self.distributed_mode = distributed_mode

    def forward(self, embeddings, target):
        if dist.is_initialized() and self.distributed_mode:
            dist.barrier()
            embeddings = all_gather_and_cat(embeddings)
            if self.use_gpu_dependent_labels:
                target = target + (target.max()+1) * dist.get_rank()
            target = all_gather_and_cat(target)

        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]).pow(2)

        negative_loss = F.relu(
            self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)

        if dist.is_initialized() and self.distributed_mode:
            loss_mult = dist.get_world_size()
        else:
            loss_mult = 1
        return loss.sum() * loss_mult