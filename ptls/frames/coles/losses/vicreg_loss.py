import torch
from torch.nn import functional as F


class VicregLoss(torch.nn.Module):
    """
    From https://github.com/facebookresearch/vicreg

    """
    def __init__(self, sim_coeff, std_coeff, cov_coeff):
        super(VicregLoss, self).__init__()

        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, model_outputs, target):
        n = len(model_outputs)
        m = len(model_outputs[0])
        ix1 = torch.arange(0, n, 2, device=model_outputs.device)
        ix2 = torch.arange(1, n, 2, device=model_outputs.device)

        assert (target[ix1] == target[ix2]).all(), "Wrong embedding positions"

        x = model_outputs[ix1]
        y = model_outputs[ix2]

        # From https://github.com/facebookresearch/vicreg:

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 +\
                   torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (n - 1)
        cov_y = (y.T @ y) / (n - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(m) +\
                   self.off_diagonal(cov_y).pow_(2).sum().div(m)

        loss = (self.sim_coeff * repr_loss +
                self.std_coeff * std_loss +
                self.cov_coeff * cov_loss)
        return loss

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()