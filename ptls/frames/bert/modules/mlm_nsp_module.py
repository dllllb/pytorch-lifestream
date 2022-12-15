import pytorch_lightning as pl
import torch
from torch.nn import BCELoss
from torchmetrics import MeanMetric

from ptls.custom_layers import StatPooling
from ptls.data_load.padded_batch import PaddedBatch
from ptls.frames.bert.losses.query_soft_max import QuerySoftmaxLoss
from ptls.nn import PBL2Norm
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder


class SequencePredictionHead(torch.nn.Module):   
    def __init__(self, embeds_dim, hidden_size=64, drop_p=0.1):
        
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(embeds_dim, hidden_size, bias=True),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_p),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid())
    def forward(self, x):
        x = self.head(x).squeeze(-1)
        return x

class MLMNSPModule(pl.LightningModule):
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
    replace_last_count:
        force replacing last n tokens of sequence
    neg_count:
        negative count for `QuerySoftmaxLoss`
    weight_mlm:
        weight of mlm loss in final loss
    weight_nsp:
        weight of nsp loss in final loss
    log_logits:
        if true than logits histogram will be logged. May be useful for `loss_temperature` tuning
    inference_pooling_strategy:
        'out' - `seq_encoder` forward (`is_reduce_requence=True`) (B, H)
        'stat' - min, max, mean, std statistics pooled from `trx_encoder` layer + 'out' from `seq_encoder` (B, H) -> (B, 5H)
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
                 replace_proba: float = 0.15,
                 replace_last_count: int = 4, 
                 neg_count: int = 1,
                 weight_mlm: float = 0.5,
                 weight_nsp: float = 0.5,
                 log_logits: bool = False,
                 inference_pooling_strategy: str = 'out'
                 ):

        super().__init__()
        self.save_hyperparameters(ignore=['trx_encoder', 'seq_encoder'])

        self.trx_encoder = trx_encoder
        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = False
        self._seq_encoder.add_cls_output = True

        self.nsp_head = SequencePredictionHead(embeds_dim=hidden_size)
        if self.hparams.norm_predict:
            self.fn_norm_predict = PBL2Norm()

        if hidden_size is None:
            hidden_size = trx_encoder.output_size

        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)

        self.loss_mlm = QuerySoftmaxLoss(temperature=loss_temperature, reduce=False)
        self.loss_nsp = BCELoss(reduce=False)

        self.train_mlm_loss = MeanMetric()
        self.valid_mlm_loss = MeanMetric()

        self.train_nsp_loss = MeanMetric()
        self.valid_nsp_loss = MeanMetric()

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

    def get_mask(self, attention_mask, seq_lens):
        last_steps = torch.arange(attention_mask.size(1), device=attention_mask.device).expand(attention_mask.size())
        last_steps = (last_steps < seq_lens[:, None]) & (last_steps > seq_lens[:, None]-self.hparams.replace_last_count-1)
        
        mask = torch.bernoulli(attention_mask.float() * self.hparams.replace_proba).bool()
        mask = torch.logical_or(mask, last_steps)
        return mask

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
        out, cls_out = self._seq_encoder(z)
        if self.hparams.norm_predict:
            out = self.fn_norm_predict(out)
        return out, cls_out

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

    def loss(self, x: PaddedBatch, y, is_train_step):
        mask = self.get_mask(x.seq_len_mask, x.seq_lens)
        masked_x = self.mask_x(x.payload, x.seq_len_mask, mask)
        out, cls_out = self.forward(PaddedBatch(masked_x, x.seq_lens))

        # MLM Part
        out = out.payload[y==1, :] # y==1 => select only true sequence pairs for MLM
        mask = mask[y==1, :]
        target = x.payload[y==1, :][mask].unsqueeze(1)  # N, 1, H
        predict = out[mask].unsqueeze(1)  # N, 1, H
        neg_ix = self.get_neg_ix(mask)
        negative = out[neg_ix[0], neg_ix[1]]  # N, nneg, H
        loss_mlm = self.loss_mlm(target, predict, negative)
        if is_train_step and self.hparams.log_logits:
            with torch.no_grad():
                logits = self.mlm_loss.get_logits(target, predict, negative)
            self.logger.experiment.add_histogram('mlm/logits',
                                                 logits.flatten().detach(), self.global_step)
        # NSP Part
        nsp_preds = self.nsp_head.forward(cls_out)
        loss_nsp = self.loss_nsp(nsp_preds, y)

        return loss_mlm, loss_nsp

    def training_step(self, batch, batch_idx):
        x_trx, y = batch
        
        z_trx = self.trx_encoder(x_trx)  # PB: B, T, H
        loss_mlm, loss_nsp = self.loss(z_trx, y, is_train_step=True)
        self.train_mlm_loss(loss_mlm)
        self.train_nsp_loss(loss_nsp)
        loss_mlm = loss_mlm.mean()
        loss_nsp = loss_nsp.mean()
        self.log(f'mlm/loss', loss_mlm)
        self.log(f'nsp/loss', loss_nsp)
        loss = self.hparams.weight_nsp*loss_nsp + self.hparams.weight_mlm*loss_mlm
        return loss

    def validation_step(self, batch, batch_idx):
        x_trx, y = batch
        z_trx = self.trx_encoder(x_trx)  # PB: B, T, H
        loss_mlm, loss_nsp = self.loss(z_trx, y, is_train_step=False)
        self.valid_nsp_loss(loss_nsp)
        self.valid_mlm_loss(loss_mlm)

    def training_epoch_end(self, _):
        self.log(f'mlm/train_mlm_loss', self.train_mlm_loss, prog_bar=False)
        self.log(f'nsp/train_nsp_loss', self.train_nsp_loss, prog_bar=False)

    def validation_epoch_end(self, _):
        self.log(f'mlm/valid_mlm_loss', self.valid_mlm_loss, prog_bar=True)
        self.log(f'nsp/valid_nsp_loss', self.valid_nsp_loss, prog_bar=False)
   
    @property
    def seq_encoder(self):
        return MLMNSPInferenceModule(pretrained_model=self)


class MLMNSPInferenceModule(torch.nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
        self.model._seq_encoder.is_reduce_sequence = True
        self.model._seq_encoder.add_cls_output = False 
        if self.model.hparams.inference_pooling_strategy=='stat':
            self.stat_pooler = StatPooling()
    def forward(self, batch):
        z_trx = self.model.trx_encoder(batch)
        out = self.model._seq_encoder(z_trx)
        if self.model.hparams.inference_pooling_strategy=='stat':
            stats = self.stat_pooler(z_trx)
            out = torch.cat([stats, out], dim=1)  # out: B, 5H
        if self.model.hparams.norm_predict:
            out = out / (out.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5)
        return out
