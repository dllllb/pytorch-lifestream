import torch
from torch import nn


class MultiLoss(nn.Module):
    def __init__(self, losses, coefs=None, warmup_disabled_losses=None):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.loss_name = "multi"
        self.restore_losses = dict()
        self.coefs = {n: 1. for n in self.losses.keys()}
        if coefs is not None:
            self.coefs.update(coefs)
        if warmup_disabled_losses is not None:
            for name in warmup_disabled_losses:
                self.restore_losses[name] = self.coefs[name]
                self.coefs[name] = 0.

    def forward(self, embs, idx=None, clusters=None):
        loss = 0
        infos = {}
        for name, l in self.losses.items():
            loss_, info_ = l(embs, idx, clusters)
            loss += self.coefs[name] * loss_
            infos[name] = info_
        infos['name'] = self.loss_name
        infos['loss'] = loss.item()
        return loss, infos

    def enable_all_losses(self):
        for name, value in self.restore_losses.items():
            self.coefs[name] = value
