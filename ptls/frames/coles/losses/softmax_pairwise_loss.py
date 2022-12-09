from torch import nn as nn
from torch.nn import functional as F
from ptls.frames.coles.sampling_strategies import PairwiseMatrixSelector


class SoftmaxPairwiseLoss(nn.Module):
    """
    Softmax Pairwise loss.
    """

    def __init__(self, pair_selector=PairwiseMatrixSelector(), temperature=0.05, eps=1e-6):
        super(SoftmaxPairwiseLoss, self).__init__()
        self.pair_selector = pair_selector
        self.temperature = temperature
        self.eps = eps

    def forward(self, embeddings, classes):
        pair_matrix = self.pair_selector.get_pair_matrix(embeddings, classes)
        similarities = F.cosine_similarity(pair_matrix[:, :, 0, :], pair_matrix[:, :, 1, :], dim=-1, eps=self.eps)
        similarities /= self.temperature
        log_matrix = (-1)*F.log_softmax(similarities)
        loss = log_matrix/(len(similarities))
        return loss[:, :1].sum()
