import pytorch_lightning as pl
import torch
from torch import nn
import warnings
from torchmetrics import MeanMetric
from typing import Tuple, Dict, List, Union

from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn import PBL2Norm
from ptls.data_load.padded_batch import PaddedBatch
from ptls.custom_layers import StatPooling, GEGLU
from ptls.nn.seq_step import LastStepEncoder

class Head(nn.Module):   
    def __init__(self, input_size, n_classes, hidden_size=64, drop_p=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_size, n_classes)
        )
    def forward(self, x):
        x = self.head(x)
        return x

class GptPretrainModule(pl.LightningModule):
    """GPT2 Language model

    Original sequence are encoded by `TrxEncoder`.
    Model `seq_encoder` predicts embedding of next transaction.
    Heads are used to predict each feature class of future transaction.

    Parameters
    ----------
    trx_encoder:
        Module for transform dict with feature sequences to sequence of transaction representations
    seq_encoder:
        Module for sequence processing. Generally this is transformer based encoder. Rnn is also possible
        Should works without sequence reduce
    head_hidden_size:
        Hidden size of heads for feature prediction
    seed_seq_len:
         Size of starting sequence without loss 
    total_steps:
        total_steps expected in OneCycle lr scheduler
    max_lr:
        max_lr of OneCycle lr scheduler
    weight_decay:
        weight_decay of Adam optimizer
    pct_start:
        % of total_steps when lr increase
    norm_predict:
        use l2 norm for transformer output or not
    inference_pooling_strategy:
        'out' - `seq_encoder` forward (`is_reduce_requence=True`) (B, H)
        'out_stat' - min, max, mean, std statistics pooled from `seq_encoder` layer (B, H) -> (B, 4H)
        'trx_stat' - min, max, mean, std statistics pooled from `trx_encoder` layer (B, H) -> (B, 4H)
        'trx_stat_out' - min, max, mean, std statistics pooled from `trx_encoder` layer + 'out' from `seq_encoder` (B, H) -> (B, 5H)
    """

    def __init__(self,
                 trx_encoder: torch.nn.Module,
                 seq_encoder: AbsSeqEncoder,
                 head_hidden_size: int = 64,
                 total_steps: int = 64000,
                 seed_seq_len: int = 16,
                 max_lr: float = 0.00005,
                 weight_decay: float = 0.0,
                 pct_start: float = 0.1,
                 norm_predict: bool = False,
                 inference_pooling_strategy: str = 'out_stat'
                 ):

        super().__init__()
        self.save_hyperparameters(ignore=['trx_encoder', 'seq_encoder'])

        self.trx_encoder = trx_encoder
        assert not self.trx_encoder.numeric_values, '`numeric_values` parameter of `trx_encoder` should be == {}. Discretize all numerical features into categorical to use Tabformer model!'
        assert self.trx_encoder.embeddings, '`embeddings` parameter for `trx_encoder` should contain at least 1 feature!'

        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = False

        self.head = nn.ModuleDict()
        for col_name, noisy_emb in self.trx_encoder.embeddings.items():
            self.head[col_name] = Head(input_size=self._seq_encoder.embedding_size, hidden_size=head_hidden_size, n_classes=noisy_emb.num_embeddings)

        if self.hparams.norm_predict:
            self.fn_norm_predict = PBL2Norm()

        self.loss = nn.CrossEntropyLoss(ignore_index=0)

        self.train_gpt_loss = MeanMetric()
        self.valid_gpt_loss = MeanMetric()

    def forward(self, batch: PaddedBatch):
        z_trx = self.trx_encoder(batch) 
        out = self._seq_encoder(z_trx)
        if self.hparams.norm_predict:
            out = self.fn_norm_predict(out)
        return out

    def loss_gpt(self, predictions, labels, is_train_step):
        loss = 0
        for col_name, head in self.head.items():
            y_pred = head(predictions[:, self.hparams.seed_seq_len:-1, :])
            y_pred = y_pred.view(-1, y_pred.size(-1))

            y_true = labels[col_name][:, self.hparams.seed_seq_len+1:]
            y_true = torch.flatten(y_true.long())

            loss += self.loss(y_pred, y_true)
        return loss

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)  # PB: B, T, H
        out = out.payload if isinstance(out, PaddedBatch) else out
        labels = batch.payload

        loss_gpt = self.loss_gpt(out, labels, is_train_step=True)
        self.train_gpt_loss(loss_gpt)
        self.log(f'gpt/loss', loss_gpt, sync_dist=True)
        return loss_gpt

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)  # PB: B, T, H
        out = out.payload if isinstance(out, PaddedBatch) else out
        labels = batch.payload

        loss_gpt = self.loss_gpt(out, labels, is_train_step=False)
        self.valid_gpt_loss(loss_gpt)

    def training_epoch_end(self, _):
        self.log(f'gpt/train_gpt_loss', self.train_gpt_loss, prog_bar=False, sync_dist=True, rank_zero_only=True)
        # self.train_gpt_loss reset not required here

    def validation_epoch_end(self, _):
        self.log(f'gpt/valid_gpt_loss', self.valid_gpt_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        # self.valid_gpt_loss reset not required here

    def configure_optimizers(self):
        optim = torch.optim.NAdam(self.parameters(),
                                 lr=self.hparams.max_lr,
                                 weight_decay=self.hparams.weight_decay,
                                 )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optim,
            max_lr=self.hparams.max_lr,
            total_steps=self.hparams.total_steps,
            pct_start=self.hparams.pct_start,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=False,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optim], [scheduler]
    
    @property
    def seq_encoder(self):
        return GPTInferenceModule(pretrained_model=self)

class GPTInferenceModule(torch.nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
        self.model.is_reduce_sequence = False

        self.stat_pooler = StatPooling()
        self.last_step = LastStepEncoder()

    def forward(self, batch):
        z_trx = self.model.trx_encoder(batch)
        out = self.model._seq_encoder(z_trx)
        out = out if isinstance(out, PaddedBatch) else PaddedBatch(out, batch.seq_lens)
        if self.model.hparams.inference_pooling_strategy=='trx_stat_out':
            stats = self.stat_pooler(z_trx)
            out = self.last_step(out)
            out = torch.cat([stats, out], dim=1)
        elif self.model.hparams.inference_pooling_strategy=='trx_stat':
            out = self.stat_pooler(z_trx)
        elif self.model.hparams.inference_pooling_strategy=='out_stat':
            out = self.stat_pooler(out)
        elif self.model.hparams.inference_pooling_strategy=='out':
            out = self.last_step(out)
        else:
            raise
        if self.model.hparams.norm_predict:
            out = out / (out.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5)
        return out