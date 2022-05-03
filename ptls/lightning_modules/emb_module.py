import torch

from ptls.lightning_modules.AbsModule import ABSModule
from ptls.metric_learn.losses import get_loss
from ptls.metric_learn.metric import BatchRecallTopPL
from ptls.metric_learn.sampling_strategies import get_sampling_strategy


class EmbModule(ABSModule):
    """pl.LightningModule for training CoLES embeddings

    Parameters
    ----------
    seq_encoder : torch.nn.Module
        sequence encoder
    head : torch.nn.Module
        head of th model
    loss : torch.nn.Module
        Loss function for training
    lr : float. Default: 1e-3
        Learning rate
    weight_decay: float. Default: 0.0
        weight decay for optimizer
    lr_scheduler_step_size : int. Default: 100
        Period of learning rate decay.
    lr_scheduler_step_gamma: float. Default: 0.1
        Multiplicative factor of learning rate decay.
    """
    def __init__(
        self,
        seq_encoder: torch.nn.Module,
        head: torch.nn.Module,
        loss: torch.nn.Module = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_step_size: int = 100,
        lr_scheduler_step_gamma: float = 0.1):

        default_loss_params = {'train.loss': 'ContrastiveLoss',
                               'train.sampling_strategy': 'HardNegativePair',
                               'train.margin': 0.5,
                               'train.neg_count': 5}
        train_params = {
            'train.lr': lr,
            'train.weight_decay': weight_decay,
            'lr_scheduler': {
                'step_size': lr_scheduler_step_size,
                'step_gamma': lr_scheduler_step_gamma
            }
        }
        if not loss:
            train_params = {**train_params, **default_loss_params}

        super().__init__(train_params, seq_encoder, loss)

        self._head = head

    @property
    def metric_name(self):
        return 'recall_top_k'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def get_loss(self):
        sampling_strategy = get_sampling_strategy(self.hparams.params)
        loss = get_loss(self.hparams.params, sampling_strategy)
        return loss

    def get_validation_metric(self):
        default_kwargs = {
            'K': 4,
            'metric': 'cosine'
        }
        if 'validation_metric_params' in self.hparams.params:
            kwargs = {**default_kwargs, **self.hparams.params['validation_metric_params']}
        else:
            kwargs = default_kwargs

        return BatchRecallTopPL(**kwargs)

    def shared_step(self, x, y):
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h, y
