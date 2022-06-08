import torch
import torchmetrics
from torch.nn import BCELoss

from ptls.lightning_modules.AbsModule import ABSModule


class SentencePairsHead(torch.nn.Module):
    def __init__(self, base_model, embeds_dim, config):
        super().__init__()
        self.base_model = base_model
        # TODO: check batch_norm position
        self.head = torch.nn.Sequential(
            torch.nn.Linear(embeds_dim * 2, config['hidden_size'], bias=True),
            torch.nn.BatchNorm1d(config['hidden_size']),
            torch.nn.ReLU(),
            torch.nn.Dropout(config['drop_p']),
            # torch.nn.BatchNorm1d(config['hidden_size']),
            torch.nn.Linear(config['hidden_size'], 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        left, right = x
        x = torch.cat([self.base_model(left), self.base_model(right)], dim=1)
        return self.head(x).squeeze(-1)


class SopNspModule(ABSModule):
    def __init__(self, validation_metric=None,
                       seq_encoder=None,
                       head=None,
                       loss=None,
                       optimizer_partial=None,
                       lr_scheduler_partial=None):

        if loss is None:
            loss = BCELoss()
        if validation_metric is None:
            validation_metric = torchmetrics.AUROC(num_classes=2, compute_on_step=False)

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        self._head = SentencePairsHead(self.seq_encoder, self.seq_encoder.embedding_size, head)

    @property
    def metric_name(self):
        return 'valid_auroc'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def shared_step(self, x, y):
        y_h = self._head(x)
        return y_h, y
