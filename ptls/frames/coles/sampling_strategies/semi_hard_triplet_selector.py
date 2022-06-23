import torch
from torch.nn import functional as F

from ptls.frames.coles.sampling_strategies.triplet_selector import TripletSelector
from ptls.frames.coles.metric import outer_pairwise_distance


class SemiHardTripletSelector(TripletSelector):
    """
        Generate triplets with semihard sampling strategy

        "FaceNet: A Unified Embedding for Face Recognition and Clustering", CVPR 2015
        https://arxiv.org/abs/1503.03832
    """

    def __init__(self, neg_count=1):
        super(SemiHardTripletSelector, self).__init__()
        self.neg_count = neg_count

    def get_triplets(self, embeddings, labels):
        n = labels.size(0)

        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        x = labels.expand(n, n) - labels.expand(n, n).t()

        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)

        m = positive_pairs.size(0)

        anchor_embed = embeddings[positive_pairs[:, 0]].detach()
        anchor_labels = labels[positive_pairs[:, 0]]

        pos_embed = embeddings[positive_pairs[:, 1]].detach()

        D_ap = F.pairwise_distance(anchor_embed, pos_embed)

        # construct matrix x (size m x n), such as x_ij == 1 <==> anchor_labels[i] == labels[j]
        x = (labels.expand(m, n) == anchor_labels.expand(n, m).t())

        mat_distances = outer_pairwise_distance(anchor_embed, embeddings.detach())  # pairwise_distance anchors x all

        neg_mat_distances = mat_distances * (x == 0).type(mat_distances.dtype)  # filter: get only negative pairs

        # negatives_outside: smallest D_an where D_an > D_ap.
        upper_bound = int((2 * n) ** 0.5) + 1
        negatives_outside = (upper_bound - neg_mat_distances) * \
                            (neg_mat_distances > D_ap.expand(n, m).t()).type(neg_mat_distances.dtype)
        values, negatives_outside = negatives_outside.topk(k=1, dim=1, largest=True)

        # negatives_inside: largest D_an
        values, negatives_inside = neg_mat_distances.topk(k=1, dim=1, largest=True)

        # whether exist negative n, such that D_an > D_ap.
        semihard_exist = ((neg_mat_distances > D_ap.expand(n, m).t()).sum(dim=1) > 0).view(-1, 1)

        negatives_indeces = torch.where(semihard_exist, negatives_outside, negatives_inside)

        triplets = torch.cat([positive_pairs, negatives_indeces], dim=1)

        return triplets