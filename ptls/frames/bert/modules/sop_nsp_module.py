import torch
import torchmetrics
from torch.nn import BCELoss

from ptls.frames.abs_module import ABSModule


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
    """Next Sequence Prediction (NSP) from [BERT](https://arxiv.org/abs/1810.04805)
    Sequences Order Prediction (SOP) from [ALBERT](https://arxiv.org/abs/1909.11942)

    Task depends on target in dataloader.

    Original sequence are split into two parts. Next `seq_encoder` makes embeddings.
    For NSP target `head` predict is a right part a correct continuation of left part.
    Correct means sampled from tha same original sequence in right order.
    Incorrect mean sampled from different original sequence.
    For SOP target `head` predict is a left and right part are in the correct order.
    Left an rights part are sampled from one sequence.

    """

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
