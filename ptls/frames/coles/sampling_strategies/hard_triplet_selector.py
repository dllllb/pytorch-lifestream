import torch

from ptls.frames.coles.sampling_strategies.triplet_selector import TripletSelector
from ptls.frames.coles.metric import outer_pairwise_distance


class HardTripletSelector(TripletSelector):
    """
        Generate triplets with all positive pairs and the neg_count hardest negative example for each anchor
    """

    def __init__(self, neg_count=1):
        super(HardTripletSelector, self).__init__()
        self.neg_count = neg_count

    def get_triplets(self, embeddings, labels):
        n = labels.size(0)

        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        x = labels.expand(n, n) - labels.expand(n, n).t()

        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)

        m = positive_pairs.size(0)

        anchor_embed = embeddings[positive_pairs[:, 0]].detach()
        anchor_labels = labels[positive_pairs[:, 0]]

        # pos_embed = embeddings[positive_pairs[:,0]].detach()

        # construct matrix x (size m x n), such as x_ij == 1 <==> anchor_labels[i] == labels[j]
        x = (labels.expand(m, n) == anchor_labels.expand(n, m).t())

        mat_distances = outer_pairwise_distance(anchor_embed, embeddings.detach())  # pairwise_distance anchors x all

        upper_bound = int((2 * n) ** 0.5) + 1
        mat_distances = ((upper_bound - mat_distances) * (x == 0).type(
            mat_distances.dtype))  # filter: get only negative pairs

        values, indices = mat_distances.topk(k=self.neg_count, dim=1, largest=True)

        triplets = torch.cat([
            positive_pairs.repeat(self.neg_count, 1),
            torch.cat(indices.unbind(dim=0)).view(-1, 1)
        ], dim=1)

        return triplets