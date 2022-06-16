import torch
from numpy.testing import assert_almost_equal

from ptls.frames.coles.metric import outer_cosine_similarity


class HistogramLoss(torch.nn.Module):
    """
    HistogramLoss

    "Learning deep embeddings with histogram loss", NIPS 2016
    https://arxiv.org/abs/1611.00822
    code based on https://github.com/valerystrizh/pytorch-histogram-loss
    """

    def __init__(self, num_steps=100):
        super(HistogramLoss, self).__init__()
        self.step = 2 / (num_steps - 1)
        self.eps = 1 / num_steps
        self.t = torch.arange(-1, 1+self.step, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]
        self.device = None

    def forward(self, embeddings, classes):
        def histogram(inds, size):

            s_repeat_ = s_repeat.clone()
            indsa = (s_repeat_floor - (self.t - self.step) > -self.eps) &  \
                    (s_repeat_floor - (self.t - self.step) < self.eps) & inds

            assert indsa.nonzero(as_tuple=False).size()[0] == size, 'Another number of bins should be used'
            zeros = torch.zeros((1, indsa.size()[1])).bool()
            zeros = zeros.to(self.device)
            indsb = torch.cat((indsa, zeros))[1:, :]
            s_repeat_[~(indsb | indsa)] = 0
            # indsa corresponds to the first condition of the second equation of the paper
            s_repeat_[indsa] = (s_repeat_ - self.t + self.step)[indsa] / self.step
            # indsb corresponds to the second condition of the second equation of the paper
            s_repeat_[indsb] = (-s_repeat_ + self.t + self.step)[indsb] / self.step

            return s_repeat_.sum(1) / size

        self.device = embeddings.device
        self.t = self.t.to(self.device)

        # L2 normalization
        classes_size = classes.size()[0]
        classes_eq = (classes.repeat(classes_size, 1)  == classes.view(-1, 1).repeat(1, classes_size)).data
        dists = outer_cosine_similarity(embeddings)

        assert ((dists > 1 + self.eps).sum().item() + (dists < -1 - self.eps).sum().item()) == 0, 'L2 normalization ' \
                                                                                                  'should be used '
        s_inds = torch.triu(torch.ones(classes_eq.size()), 1).bool()
        s_inds = s_inds.to(self.device)
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq[s_inds].sum().item()
        neg_size = (~classes_eq[s_inds]).sum().item()
        s = dists[s_inds].view(1, -1)
        s = s.clamp(-1 + 1e-6, 1 - 1e-6)
        s_repeat = s.repeat(self.tsize, 1)
        s_repeat_floor = (torch.floor((s_repeat.data + 1.0 - 1e-6) / self.step) * self.step - 1.0).float()

        histogram_pos = histogram(pos_inds, pos_size)
        assert_almost_equal(histogram_pos.sum().item(), 1, decimal=1,
                            err_msg='Not good positive histogram', verbose=True)
        histogram_neg = histogram(neg_inds, neg_size)
        assert_almost_equal(histogram_neg.sum().item(), 1, decimal=1,
                            err_msg='Not good negative histogram', verbose=True)
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0])
        histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).bool()
        histogram_pos_inds = histogram_pos_inds.to(self.device)
        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        loss = torch.sum(histogram_neg * histogram_pos_cdf)

        return loss