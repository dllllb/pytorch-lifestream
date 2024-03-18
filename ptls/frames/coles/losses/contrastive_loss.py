import torch
from torch import nn as nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss

    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(self, margin, sampling_strategy):
        super().__init__()
        self.margin = margin
        self.pair_selector = sampling_strategy
        self.loss_name = "coles"

    def forward(self, embeddings, target, *argv):
        if type(target) is dict:
            target = target["coles_target"]

        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]).pow(2)

        negative_loss = F.relu(
            self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0).mean()

        return loss, {"loss_0": loss.cpu().item()}


class MultiContrastiveLoss(nn.Module):
    def __init__(self, margin, sampling_strategy):
        super().__init__()
        self.margin = margin
        self.pair_selector = sampling_strategy
        self.loss_name = "coles"

    def forward(self, multi_embeddings, target, *argv):
        if type(target) is dict:
            target = target["coles_target"]

        loss = 0
        info = dict()
        for i, embeddings in enumerate(multi_embeddings):
            positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
            positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]],
                                                embeddings[positive_pairs[:, 1]]).pow(2)

            negative_loss = F.relu(self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]],
                                                                     embeddings[negative_pairs[:, 1]])).pow(2)
            loss_i = torch.cat([positive_loss, negative_loss], dim=0).mean()
            loss += loss_i
            info["loss_" + str(i)] = loss_i.cpu().item()
        loss = loss / len(multi_embeddings)
        return loss, info
