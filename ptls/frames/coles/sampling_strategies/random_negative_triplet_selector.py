import torch

from ptls.frames.coles.sampling_strategies.triplet_selector import TripletSelector


class RandomNegativeTripletSelector(TripletSelector):
    """
        Generate triplets with all positive pairs and random negative example for each anchor
    """

    def __init__(self, neg_count=1):
        super(RandomNegativeTripletSelector, self).__init__()
        self.neg_count = neg_count

    def get_triplets(self, embeddings, labels):
        n = labels.size(0)

        # construct matrix x, such as x_ij == 1 <==> labels[i] == labels[j]
        x = labels.expand(n, n) - labels.expand(n, n).t()

        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)

        m = positive_pairs.size(0)
        anchor_labels = labels[positive_pairs[:, 0]]

        # construct matrix x (size m x n), such as x_ij == 1 <==> anchor_labels[i] == labels[j]
        x = (labels.expand(m, n) == anchor_labels.expand(n, m).t())

        negative_pairs = (x == 0).type(embeddings.dtype)
        negative_pairs_prob = (negative_pairs.t() / negative_pairs.sum(dim=1)).t()
        negative_pairs_rand = torch.multinomial(negative_pairs_prob, 1)

        triplets = torch.cat([positive_pairs, negative_pairs_rand], dim=1)

        return triplets