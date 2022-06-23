import torch


class QuerySoftmaxLoss(torch.nn.Module):
    def __init__(self, temperature: float = 1.0, reduce: bool = True):
        """

        Parameters
        ----------
        temperature:
            softmax(logits * temperature)
        reduce:
            if `reduce` then `loss.mean()` returned. Else loss by elements returned
        """

        super().__init__()
        self.temperature = temperature
        self.reduce = reduce

    def forward(self, anchor, pos, neg):
        logits = self.get_logits(anchor, pos, neg)
        probas = torch.softmax(logits, dim=1)
        loss = -torch.log(probas[:, 0])
        if self.reduce:
            return loss.mean()
        return loss

    def get_logits(self, anchor, pos, neg):
        all_counterparty = torch.cat([pos, neg], dim=1)
        logits = (anchor * all_counterparty).sum(dim=2) * self.temperature
        return logits