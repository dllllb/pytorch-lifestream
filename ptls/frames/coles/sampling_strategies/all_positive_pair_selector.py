import torch

from ptls.frames.coles.sampling_strategies.pair_selector import PairSelector


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """

    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        n = labels.size(0)
        x = labels.expand(n, n) - labels.expand(n, n).t()

        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)
        negative_pairs = torch.triu((x != 0).int(), diagonal=1).nonzero(as_tuple=False)

        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs