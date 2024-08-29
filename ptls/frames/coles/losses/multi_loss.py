import torch
from torch import nn


class MultiLoss(nn.Module):
    def __init__(self, losses, coefs=None):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.loss_name = "multi"
        self.coefs = {n: 1. for n in self.losses.keys()}
        if coefs is not None:
            self.coefs.update(coefs)

    def forward(self, embs, idx=None, clusters=None):
        loss = 0
        info = {}
        for name, l in self.losses.items():
            loss_, info_ = l(embs, idx, clusters)
            loss += self.coefs[name] * loss_
            info.update(info_)
        info['total_loss'] = loss.item()
        return loss, info
