import torch
from torch import nn as nn
from torch.nn import functional as F


class BinomialDevianceLoss(nn.Module):
    """
    Binomial Deviance loss

    "Deep Metric Learning for Person Re-Identification", ICPR2014
    https://arxiv.org/abs/1407.4979
    """

    def __init__(self, pair_selector, alpha=1, beta=1, C=1):
        super(BinomialDevianceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.C = C
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):

        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)

        pos_pair_similarity = F.cosine_similarity(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]], dim=1)
        neg_pair_similarity = F.cosine_similarity(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]], dim=1)

        pos_loss = torch.mean(torch.log(1 + torch.exp(-self.alpha*(pos_pair_similarity - self.beta))))
        neg_loss = torch.mean(torch.log(1 + torch.exp(self.alpha*self.C*(neg_pair_similarity - self.beta))))

        res_loss = (pos_loss + neg_loss) * (len(target))

        return res_loss