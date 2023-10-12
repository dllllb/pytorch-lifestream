import torchmetrics


class AccuracyMetric(torchmetrics.MeanMetric):
    def __init__(self):
        super().__init__()

    def update(self, preds, target):
        super().update((preds.round() == target).mean())
