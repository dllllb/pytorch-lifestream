import torch
from torch import nn as nn
from torch.nn import functional as F
from ptls.frames.coles.sampling_strategies import MatrixMasker


class SoftmaxLoss(nn.Module):
    """
    Softmax loss.
    """

    def __init__(self, masker=None, eps=1e-6, temperature=0.05):
        super(SoftmaxLoss, self).__init__()
        self.masker = masker
        if masker is None:
            self.masker = MatrixMasker()
        self.eps = eps
        self.temperature = temperature

    def forward(self, embeddings, classes):
        similarities = self.get_sim_matrix(embeddings, embeddings, eps=self.eps)
        similarities /= self.temperature
        log_matrix = (-1)*F.log_softmax(similarities)
        masked_matrix = self.masker.get_masked_matrix(log_matrix, classes)
        loss = masked_matrix/len(similarities)
        return loss.sum()

    @staticmethod
    def get_sim_matrix(a, b, eps):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt
