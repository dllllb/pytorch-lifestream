import numpy as np
import torch
import pytorch_lightning as pl

from omegaconf import DictConfig
from transformers import LongformerConfig, LongformerModel

from ptls.trx_encoder import PaddedBatch
from torchmetrics import MeanMetric


class QuerySoftmaxLoss(torch.nn.Module):
    def __init__(self, temperature: float = 1.0, reduce: bool = True):
        """

        Parameters
        ----------
        temperature:
            softmax(logits * temperature)
        reduce:
            if `reduce` then `loss.mean()` returned. Else loss by elements returned
        """

        super().__init__()
        self.temperature = temperature
        self.reduce = reduce

    def forward(self, anchor, pos, neg):
        logits = self.get_logits(anchor, pos, neg)
        probas = torch.softmax(logits, dim=1)
        loss = -torch.log(probas[:, 0])
        if self.reduce:
            return loss.mean()
        return loss

    def get_logits(self, anchor, pos, neg):
        all_counterparty = torch.cat([pos, neg], dim=1)
        logits = (anchor * all_counterparty).sum(dim=2) * self.temperature
        return logits


class MLMPretrainModule(pl.LightningModule):
    def __init__(self,
                 trx_encoder: torch.nn.Module,
                 hidden_size: int,
                 loss_temperature: float,
                 total_steps: int,
                 max_lr: float = 0.001,
                 weight_decay: float = 0.0,
                 pct_start: float = 0.1,
                 norm_predict: bool = True,
                 num_attention_heads: int = 1,
                 intermediate_size: int = 128,
                 num_hidden_layers: int = 1,
                 attention_window: int = 16,
                 max_position_embeddings: int = 4000,
                 replace_proba: float = 0.1,
                 neg_count: int = 1,
                 log_logits: bool = False,
                 ):
        """

        Parameters
        ----------
        trx_encoder:
            Module for transform dict with feature sequences to sequence of transaction representations
        hidden_size:
            Output size of `trx_encoder`. Hidden size of internal transformer representation
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
        num_attention_heads:
            parameter for Longformer
        intermediate_size:
            parameter for Longformer
        num_hidden_layers:
            parameter for Longformer
        attention_window:
            parameter for Longformer
        max_position_embeddings:
            parameter for Longformer
        replace_proba:
            probability of masking transaction embedding
        neg_count:
            negative count for `QuerySoftmaxLoss`
        log_logits:
            if true than logits histogram will be logged. May be useful for `loss_temperature` tuning
        """

        super().__init__()
        self.save_hyperparameters()

        self.trx_encoder = trx_encoder

        self.token_cls = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)

        self.transf = LongformerModel(
            config=LongformerConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_hidden_layers,
                vocab_size=4,
                max_position_embeddings=self.hparams.max_position_embeddings,
                attention_window=attention_window,
            ),
            add_pooling_layer=False,
        )

        self.loss_fn = QuerySoftmaxLoss(temperature=loss_temperature, reduce=False)

        self.train_mlm_loss = MeanMetric(compute_on_step=False)
        self.valid_mlm_loss = MeanMetric(compute_on_step=False)

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
        B, T, H = z.payload.size()
        device = z.payload.device

        if self.training:
            start_pos = np.random.randint(0, self.hparams.max_position_embeddings - T - 1, 1)[0]
        else:
            start_pos = 0

        inputs_embeds = z.payload
        attention_mask = z.seq_len_mask.float()

        inputs_embeds = torch.cat([
            self.token_cls.expand(inputs_embeds.size(0), 1, H),
            inputs_embeds,
        ], dim=1)
        attention_mask = torch.cat([
            torch.ones(inputs_embeds.size(0), 1, device=device),
            attention_mask,
        ], dim=1)
        position_ids = torch.arange(T + 1, device=z.device).view(1, -1).expand(B, T + 1) + start_pos
        global_attention_mask = torch.cat([
            torch.ones(inputs_embeds.size(0), 1, device=device),
            torch.zeros(inputs_embeds.size(0), inputs_embeds.size(1) - 1, device=device),
        ], dim=1)

        out = self.transf(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            global_attention_mask=global_attention_mask,
        ).last_hidden_state

        if self.hparams.norm_predict:
            out = out / (out.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5)

        return PaddedBatch(out[:, 1:], z.seq_lens)

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
