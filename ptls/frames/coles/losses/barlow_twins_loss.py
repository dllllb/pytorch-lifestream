import torch


class BarlowTwinsLoss(torch.nn.Module):
    """
    From https://github.com/facebookresearch/barlowtwins

    """
    def __init__(self, lambd):
        super().__init__()

        self.lambd = lambd

    def forward(self, model_outputs, target):
        n = len(model_outputs)
        ix1 = torch.arange(0, n, 2, device=model_outputs.device)
        ix2 = torch.arange(1, n, 2, device=model_outputs.device)

        assert (target[ix1] == target[ix2]).all(), "Wrong embedding positions"

        z1 = model_outputs[ix1]
        z2 = model_outputs[ix2]

        # empirical cross-correlation matrix
        c = torch.mm(z1.T, z2)
        c.div_(n // 2)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()