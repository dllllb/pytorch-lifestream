import torch
from torch import nn


class MultiLoss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.loss_name = "multi"

    def forward(self, embs, idx=None, clusters=None):
        loss = 0
        infos = {}
        for name, l in self.losses.items():
            loss_, info_ = l(embs, idx, clusters)
            loss += loss_
            infos[name] = info_
        infos['name'] = self.loss_name
        infos['loss'] = loss.item()
        return loss, infos
