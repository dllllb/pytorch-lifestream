import torchmetrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score


metrics = {
    'rocauc': roc_auc_score,
    'recall': recall_score,
    'precision': precision_score,
    'f1': f1_score,
    'accuracy': accuracy_score
}


class SKLMetric(torchmetrics.MeanMetric):
    def __init__(self, metric_name='rocauc'):
        assert metric_name in ['rocauc', 'f1', 'recall', 'precision', 'accuracy']
        super().__init__()
        self.metric = metrics[metric_name]()

    def update(self, preds, target):
        super().update(self.metric(target, preds))
