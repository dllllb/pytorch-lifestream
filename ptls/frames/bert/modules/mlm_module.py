import pytorch_lightning as pl
import torch
from torchmetrics import MeanMetric

from ptls.frames.bert.losses.query_soft_max import QuerySoftmaxLoss
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn import PBL2Norm
from ptls.data_load.padded_batch import PaddedBatch


class MLMPretrainModule(pl.LightningModule):
    """Masked Language Model (MLM) from [ROBERTA](https://arxiv.org/abs/1907.11692)

    Original sequence are encoded by `TrxEncoder`.
    Randomly sampled trx representations are replaced by MASK embedding.
    Transformer `seq_encoder` reconstruct masked embeddings.
    The loss function tends to make closer trx embedding and his predict.
    Negative samples are used to avoid trivial solution.

    Parameters
    ----------
    trx_encoder:
        Module for transform dict with feature sequences to sequence of transaction representations
    seq_encoder:
        Module for sequence processing. Generally this is transformer based encoder. Rnn is also possible
        Should works without sequence reduce
    hidden_size:
        Size of trx_encoder output.
    loss_temperature:
         temperature parameter of `QuerySoftmaxLoss`
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
    replace_proba:
        probability of masking transaction embedding
    neg_count:
        negative count for `QuerySoftmaxLoss`
    log_logits:
        if true than logits histogram will be logged. May be useful for `loss_temperature` tuning
    """

    def __init__(self,
                 trx_encoder: torch.nn.Module,
                 seq_encoder: AbsSeqEncoder,
                 total_steps: int,
                 hidden_size: int = None,
                 loss_temperature: float = 20.0,
                 max_lr: float = 0.001,
                 weight_decay: float = 0.0,
                 pct_start: float = 0.1,
                 norm_predict: bool = True,
                 replace_proba: float = 0.1,
                 neg_count: int = 1,
                 log_logits: bool = False,
                 ):

        super().__init__()
        self.save_hyperparameters(ignore=['trx_encoder', 'seq_encoder'])

        self.trx_encoder = trx_encoder
        self.seq_encoder = seq_encoder
        self.seq_encoder.is_reduce_sequence = False

        if self.hparams.norm_predict:
            self.fn_norm_predict = PBL2Norm()

        if hidden_size is None:
            hidden_size = trx_encoder.output_size

        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)

        self.loss_fn = QuerySoftmaxLoss(temperature=loss_temperature, reduce=False)

        self.train_mlm_loss = MeanMetric()
        self.valid_mlm_loss = MeanMetric()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),
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

    def get_mask(self, attention_mask):
        return torch.bernoulli(attention_mask.float() * self.hparams.replace_proba).bool()

    def mask_x(self, x, attention_mask, mask):
        shuffled_tokens = x[attention_mask.bool()]
        B, T, H = x.size()
        ix = torch.multinomial(torch.ones(shuffled_tokens.size(0)), B * T, replacement=True)
        shuffled_tokens = shuffled_tokens[ix].view(B, T, H)

        rand = torch.rand(B, T, device=x.device).unsqueeze(2).expand(B, T, H)
        replace_to = torch.where(
            rand < 0.8,
            self.token_mask.expand_as(x),  # [MASK] token 80%
            torch.where(
                rand < 0.9,
                shuffled_tokens,  # random token 10%
                x,  # unchanged 10%
            )
        )
        return torch.where(mask.bool().unsqueeze(2).expand_as(x), replace_to, x)

    def forward(self, z: PaddedBatch):
        out = self.seq_encoder(z)
        if self.hparams.norm_predict:
            out = self.fn_norm_predict(out)
        return out

    def get_neg_ix(self, mask):
        """Sample from predicts, where `mask == True`, without self element.
        sample from predicted tokens from batch
        """
        mask_num = mask.int().sum()
        mn = 1 - torch.eye(mask_num, device=mask.device)
        neg_ix = torch.multinomial(mn, self.hparams.neg_count)

        b_ix = torch.arange(mask.size(0), device=mask.device).view(-1, 1).expand_as(mask)[mask][neg_ix]
        t_ix = torch.arange(mask.size(1), device=mask.device).view(1, -1).expand_as(mask)[mask][neg_ix]
        return b_ix, t_ix

    def loss_mlm(self, x: PaddedBatch, is_train_step):
        mask = self.get_mask(x.seq_len_mask)
        masked_x = self.mask_x(x.payload, x.seq_len_mask, mask)

        out = self.forward(PaddedBatch(masked_x, x.seq_lens)).payload

        target = x.payload[mask].unsqueeze(1)  # N, 1, H
        predict = out[mask].unsqueeze(1)  # N, 1, H
        neg_ix = self.get_neg_ix(mask)
        negative = out[neg_ix[0], neg_ix[1]]  # N, nneg, H
        loss = self.loss_fn(target, predict, negative)

        if is_train_step and self.hparams.log_logits:
            with torch.no_grad():
                logits = self.loss_fn.get_logits(target, predict, negative)
            self.logger.experiment.add_histogram('mlm/logits',
                                                 logits.flatten().detach(), self.global_step)
        return loss

    def training_step(self, batch, batch_idx):
        x_trx = batch
        z_trx = self.trx_encoder(x_trx)  # PB: B, T, H
        loss_mlm = self.loss_mlm(z_trx, is_train_step=True)
        self.train_mlm_loss(loss_mlm)
        loss_mlm = loss_mlm.mean()
        self.log(f'mlm/loss', loss_mlm)
        return loss_mlm

    def validation_step(self, batch, batch_idx):
        x_trx = batch
        z_trx = self.trx_encoder(x_trx)  # PB: B, T, H
        loss_mlm = self.loss_mlm(z_trx, is_train_step=False)
        self.valid_mlm_loss(loss_mlm)

    def training_epoch_end(self, _):
        self.log(f'mlm/train_mlm_loss', self.train_mlm_loss, prog_bar=False)

    def validation_epoch_end(self, _):
        self.log(f'mlm/valid_mlm_loss', self.valid_mlm_loss, prog_bar=True)
