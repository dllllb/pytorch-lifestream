import torchmetrics


class CpcAccuracy(torchmetrics.MeanMetric):
    def __init__(self, loss):
        super().__init__()

        self.loss = loss

    def update(self, *input):
        super().update(self.loss.cpc_accuracy(*input))
