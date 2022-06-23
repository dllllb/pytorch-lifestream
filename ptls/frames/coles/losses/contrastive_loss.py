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
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = sampling_strategy

    def forward(self, embeddings, target):

        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]).pow(2)

        negative_loss = F.relu(
            self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)

        return loss.sum()