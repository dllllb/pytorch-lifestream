import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from ptls.frames.coles.losses.dist_utils import all_gather_and_cat


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss

    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(self, margin, sampling_strategy, distributed_mode=False, do_loss_mult=False):
        super().__init__()
        self.margin = margin
        self.pair_selector = sampling_strategy
        self.distributed_mode = distributed_mode
        self.do_loss_mult = do_loss_mult
        self.loss_name = "coles"

    def forward(self, embeddings, target, i=0):
        if type(target) is dict:
            target = target["coles_target"]
        if dist.is_initialized() and self.distributed_mode:
            dist.barrier()
            embeddings = all_gather_and_cat(embeddings)
            target = target + (target.max()+1) * dist.get_rank()
            target = all_gather_and_cat(target)

        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]).pow(2)

        negative_loss = F.relu(
            self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0).mean()

        if dist.is_initialized() and self.do_loss_mult:
            loss_mult = dist.get_world_size()
        else:
            loss_mult = 1
        return loss * loss_mult, {f'COLES_loss_{i}': loss.item()}


class MultiContrastiveLoss(ContrastiveLoss):
    def forward(self, list_embeddings, target, *argv):
        all_info = dict()
        total_loss = 0
        for i, embs in enumerate(list_embeddings):
            loss, info = super().forward(embs, target, i)
            total_loss += loss
            all_info.update(info)
        return total_loss, all_info


class CLUBLoss(nn.Module):
    def embed_loss_prob(self, pos_probs, neg_probs):
        loss = pos_probs.mean() - neg_probs.mean()
        with torch.no_grad():
            acc = ((pos_probs > 0.5).sum() + (neg_probs < 0.5).sum()) / (pos_probs.shape[0] + neg_probs.shape[0])
            pos_mean = pos_probs.mean()
            pos_std = pos_probs.std()
            pos_delta = pos_probs.max() - pos_probs.min()
            neg_mean = neg_probs.mean()
            neg_std = neg_probs.std()
            neg_delta = neg_probs.max() - neg_probs.min()
        return loss, {"CLUB_embed_loss": loss.item(),
                      "CLUB_accuracy": acc.item(),
                      "CLUB_pos_pred_mean": pos_mean.item(),
                      "CLUB_neg_pred_mean": neg_mean.item(),
                      "CLUB_pos_pred_std": pos_std.item(),
                      "CLUB_neg_pred_std": neg_std.item(),
                      "CLUB_pos_pred_delta": pos_delta.item(),
                      "CLUB_neg_pred_delta": neg_delta.item()}

    def pred_loss_prob(self, pos_probs, neg_probs):
        pos_probs = - pos_probs.mean()
        loss = pos_probs + neg_probs.mean()
        return loss, {"CLUB_prob_loss": pos_probs.item()}
