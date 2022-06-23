import torch
from torch.nn import functional as F


class MarginLoss(torch.nn.Module):

    """
    Margin loss

    "Sampling Matters in Deep Embedding Learning", ICCV 2017
    https://arxiv.org/abs/1706.07567

    """

    def __init__(self, pair_selector, margin=1, beta=1.2):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.beta = beta
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):

        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)

        d_ap = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]])
        d_an = F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])

        pos_loss = torch.clamp(d_ap - self.beta + self.margin, min=0.0)
        neg_loss = torch.clamp(self.beta - d_an + self.margin, min=0.0)

        loss = torch.cat([pos_loss, neg_loss], dim=0)

        return loss.sum()