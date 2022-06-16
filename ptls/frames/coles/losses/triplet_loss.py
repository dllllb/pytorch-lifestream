from torch import nn as nn
from torch.nn import functional as F


class TripletLoss(nn.Module):
    """
    Triplets loss

    "Deep metric learning using triplet network", SIMBAD 2015
    https://arxiv.org/abs/1412.6622
    """

    def __init__(self, margin, triplet_selector):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = F.pairwise_distance(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]])
        an_distances = F.pairwise_distance(embeddings[triplets[:, 0]], embeddings[triplets[:, 2]])
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.sum()