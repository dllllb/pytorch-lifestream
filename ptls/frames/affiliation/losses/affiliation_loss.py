import torch
from torch import nn


class AffiliationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_head = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.LazyInstanceNorm1d(),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=1),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()

    def forward(self, long_embs, short_embs, labels):
        n = int(short_embs.shape[0] / long_embs.shape[0])
        long_embs = torch.tile(long_embs, (n, 1))
        preds = self.loss_head(torch.cat([long_embs, short_embs], dim=-1))
        loss = self.criterion(preds, labels)
        return loss, {'preds': preds.detach()}
