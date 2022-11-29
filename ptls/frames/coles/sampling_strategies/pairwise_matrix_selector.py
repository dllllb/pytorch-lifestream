import torch

from ptls.frames.coles.sampling_strategies.all_positive_pair_selector import AllPositivePairSelector


class PairwiseMatrixSelector(AllPositivePairSelector):
    """
    TODO
    """
    def __init__(self, pos_proportion=0.3, neg_proportion=1.0):
        super(PairwiseMatrixSelector, self).__init__()
        self.balance = False
        self.pos_proportion = pos_proportion
        self.neg_proportion = neg_proportion

    def get_pair_matrix(self, embeddings, classes):
        positive_pairs, negative_pairs = self.get_pairs(embeddings, classes)
        inds = torch.tensor([], device=classes.device).long()
        for ind in torch.randperm(len(embeddings))[:int(self.pos_proportion*len(embeddings))]:
            neg_pair_inds = torch.cat((torch.where(negative_pairs[:, 0] == ind)[0], torch.where(negative_pairs[:, 1] == ind)[0]))
            if self.neg_proportion < 1.0:
                rand_inds = torch.randperm(len(neg_pair_inds))[:int(self.neg_proportion*len(neg_pair_inds))]
                neg_pair_inds = neg_pair_inds[rand_inds]
            pos_pair_inds = torch.cat((torch.where(positive_pairs[:, 0] == ind)[0], torch.where(positive_pairs[:, 1] == ind)[0]))
            pos_pair_inds = pos_pair_inds.view((-1, 1))
            pairs_inds = torch.cat((positive_pairs[pos_pair_inds], torch.unsqueeze(negative_pairs[neg_pair_inds], 0).repeat((len(pos_pair_inds), 1, 1))), dim=1)
            inds = torch.cat((inds, pairs_inds), 0)
        return embeddings[inds]
