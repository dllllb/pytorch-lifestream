import torch

from ptls.frames.coles.sampling_strategies.pair_selector import PairSelector
from ptls.frames.coles.metric import outer_pairwise_distance


class HardNegativePairSelector(PairSelector):
    """
    Generates all possible possitive pairs given labels and
         neg_count hardest negative example for each example
    """

    def __init__(self, neg_count=1):
        super(HardNegativePairSelector, self).__init__()
        self.neg_count = neg_count

    def get_pairs(self, embeddings, labels):
        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        n = labels.size(0)
        x = labels.expand(n, n) - labels.expand(n, n).t()

        # positive pairs
        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)

        # hard negative minning
        mat_distances = outer_pairwise_distance(embeddings.detach())  # pairwise_distance

        upper_bound = int((2 * n) ** 0.5) + 1
        mat_distances = ((upper_bound - mat_distances) * (x != 0).type(
            mat_distances.dtype))  # filter: get only negative pairs

        values, indices = mat_distances.topk(k=self.neg_count, dim=0, largest=True)
        negative_pairs = torch.stack([
            torch.arange(0, n, dtype=indices.dtype, device=indices.device).repeat(self.neg_count),
            torch.cat(indices.unbind(dim=0))
        ]).t()

        return positive_pairs, negative_pairs