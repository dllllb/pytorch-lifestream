from torch import nn


class AffiliationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()

    def forward(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss
