import torch
from torch.nn import BCELoss

from ptls.lightning_modules.AbsModule import ABSModule
from ptls.seq_to_target import EpochAuroc


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
    def __init__(self, params):
        super().__init__(params)

        self._head = SentencePairsHead(self.seq_encoder, self.seq_encoder.embedding_size, params['head'])

    @property
    def metric_name(self):
        return 'valid_auroc'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def get_loss(self):
        return BCELoss()

    def get_validation_metric(self):
        return EpochAuroc()

    def shared_step(self, x, y):
        y_h = self._head(x)
        return y_h, y
