import pytorch_lightning as pl
import torch
from torch import nn
import warnings
from torchmetrics import MeanMetric
from typing import Tuple, Dict

from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn import PBL2Norm
from ptls.data_load.padded_batch import PaddedBatch
from ptls.custom_layers import StatPooling


class TabformerPretrainModule(pl.LightningModule):
    """Tabformer Model (MLM) from [Tabular Transformers for Modeling Multivariate Time Series](https://arxiv.org/abs/2011.01843)

    Original sequence are encoded by `TrxEncoder`.
    Randomly sampled features are replaced by MASK token.
    Transformer `seq_encoder` reconstruct masked tokens.
    The loss function is a classification loss.

    Parameters
    ----------
    trx_encoder:
        Module for transform dict with feature sequences to sequence of transaction representations
    feature_encoder:
        Module that apply transformer layers to output of trx_encoder on feature dimension, (B, T, H_in) -> (B, T, H_out)
    seq_encoder:
        Module for sequence processing. Generally this is transformer based encoder. Rnn is also possible
        Should works without sequence reduce
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
    mask_prob:
        probability of masking randomly selected feature
    inference_pooling_strategy:
        'out' - `seq_encoder` forward (`is_reduce_requence=True`) (B, H)
        'stat' - min, max, mean, std statistics pooled from `trx_encoder` layer + 'out' from `seq_encoder` (B, H) -> (B, 5H)
    """

    def __init__(self,
                 trx_encoder: torch.nn.Module,
                 feature_encoder: torch.nn.Module,
                 seq_encoder: AbsSeqEncoder,
                 total_steps: int,
                 max_lr: float = 0.001,
                 weight_decay: float = 0.0,
                 pct_start: float = 0.1,
                 norm_predict: bool = False,
                 mask_prob: float = 0.15,
                 inference_pooling_strategy: str = 'out'
                 ):

        super().__init__()
        self.save_hyperparameters(ignore=['trx_encoder', 'seq_encoder', 'feature_encoder'])

        self.trx_encoder = trx_encoder
        self.feature_encoder = feature_encoder

        assert not self.trx_encoder.custom_embeddings, '`custom_embeddings` parameter of `trx_encoder` should be == {}. Discretize all numerical features into categorical to use Tabformer model!'
        noisy_embeds = list(self.trx_encoder.embeddings.values())
        assert noisy_embeds, '`embeddings` parameter for `trx_encoder` should contain at least 1 feature!'
        self.feature_emb_dim = noisy_embeds[0].embedding_dim
        self.num_f = len(self.trx_encoder.embeddings)

        self.head = nn.ModuleList()
        in_head_dim = self.feature_emb_dim * self.num_f
        for n_emb in noisy_embeds:
            self.head += [nn.Linear(in_head_dim, n_emb.num_embeddings+1)]
            assert all(n_emb.embedding_dim == self.feature_emb_dim for n_emb in noisy_embeds), 'Out dimensions for all features in `embeddings` parameter of `trx_encoder` should be equal for Tabformer model!'

        warnings.warn("With Tabformer model set `in` value in `embeddings`parameter of `trx_encoder` equal to actual number of unique feature values + 1")

        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = False

        if self.hparams.norm_predict:
            self.fn_norm_predict = PBL2Norm()

        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, self.feature_emb_dim), requires_grad=True)

        self.loss = nn.CrossEntropyLoss()

        self.train_tabformer_loss = MeanMetric()
        self.valid_tabformer_loss = MeanMetric()
        self.mask_prob = mask_prob

        self.lin_proj = nn.Sequential(nn.Linear(self.feature_emb_dim, self.feature_emb_dim * self.num_f),
                                      nn.GELU(),
                                      nn.LayerNorm(self.feature_emb_dim * self.num_f, eps=1e-12)
                                      )

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

    def forward(self, z: PaddedBatch):
        out = self._seq_encoder(z)
        if self.hparams.norm_predict:
            out = self.fn_norm_predict(out)
        return out

    def get_masks_and_labels(self, batch: PaddedBatch) -> Tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        feature_tensors, random_words = [], []
        for field_name, noisy_emb_module in self.trx_encoder.embeddings.items():
            feature_tensors += [batch.payload[field_name].unsqueeze(0)]
            random_words += [noisy_emb_module(torch.randint(1, noisy_emb_module.num_embeddings, batch.seq_len_mask.shape, dtype=torch.long).to(batch.device)).unsqueeze(0)]
        feature_tensors = torch.cat(feature_tensors)
        random_words = torch.cat(random_words).permute(1, 2, 0, 3)
        pad_mask_tensors = batch.seq_len_mask.unsqueeze(0).repeat((feature_tensors.shape[0], 1, 1))

        pad_tokens_mask = ~pad_mask_tensors.type(torch.bool)
        probability_matrix = torch.full(pad_tokens_mask.shape, self.mask_prob).to(batch.device)
        probability_matrix.masked_fill_(pad_tokens_mask, value=0.0)

        return self.tabformer_mask(feature_tensors, probability_matrix) + (random_words,)

    def tabformer_mask(self, inputs: torch.Tensor, probability_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            labels = inputs.clone()
            # We sample a few tokens in each sequence for masked-LM training (with probability defined in probability_matrix
            # defaults to 0.15 in Bert/RoBERTa)
            masked_indices = torch.bernoulli(probability_matrix).bool().to(inputs.device)
            labels[~masked_indices] = -100  # We only compute loss on masked tokens
    
            # 80% of the time, we replace masked input tokens with [MASK] token
            indices_replaced = (torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices)
    
            # 10% of the time, we replace masked input tokens with random word
            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            indices_random = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(inputs.device) & masked_indices & ~indices_replaced)
    
            return labels.permute(1, 2, 0), masked_indices.permute(1, 2, 0), indices_replaced.permute(1, 2, 0)

    def loss_tabformer(self, x: PaddedBatch, target, is_train_step):
        out = self.forward(x)
        sequence_output = out.payload
        masked_lm_labels = target.view(-1, self.num_f).permute(1, 0)  # (B, T, NUM_F) -> (NUM_F, B*T)

        expected_sz = (-1, self.num_f, self.feature_emb_dim)
        sequence_output = sequence_output.reshape(expected_sz).permute(1, 0, 2)  # (B, T, NUM_F*H) -> (NUM_F, B*T, H)
        sequence_output = self.lin_proj(sequence_output)

        seq_out_feature = torch.chunk(sequence_output, self.num_f, dim=0)
        labels_feature = torch.chunk(masked_lm_labels, self.num_f, dim=0)

        loss = 0
        for f_ix in range(self.num_f):
            seq_out_f, labels_f = seq_out_feature[f_ix].squeeze(0), labels_feature[f_ix].squeeze(0)
            out_f = self.head[f_ix](seq_out_f)
            loss_f = self.loss(out_f, labels_f)
            loss += loss_f
        return loss

    def training_step(self, batch, batch_idx):
        tabf_labels, MASK_token_mask, RANDOM_token_mask, random_words = self.get_masks_and_labels(batch)
        z_trx = self.trx_encoder(batch)  # PB: B, T, H
        
        payload = z_trx.payload.view(z_trx.payload.shape[:-1] + (-1, self.feature_emb_dim))
        payload[MASK_token_mask] = self.token_mask
        payload[RANDOM_token_mask] = random_words[RANDOM_token_mask]
        payload = self.feature_encoder(payload)

        z_trx._payload = payload

        loss_tabformer = self.loss_tabformer(z_trx, tabf_labels, is_train_step=True)
        self.train_tabformer_loss(loss_tabformer)
        self.log(f'tabformer/loss', loss_tabformer)
        return loss_tabformer

    def validation_step(self, batch, batch_idx):
        tabf_labels, MASK_token_mask, RANDOM_token_mask, random_words = self.get_masks_and_labels(batch)
        z_trx = self.trx_encoder(batch)  # PB: B, T, H

        payload = z_trx.payload.view(z_trx.payload.shape[:-1] + (-1, self.feature_emb_dim))
        payload[MASK_token_mask] = self.token_mask
        payload[RANDOM_token_mask] = random_words[RANDOM_token_mask]
        payload = self.feature_encoder(payload)

        z_trx._payload = payload

        loss_tabformer = self.loss_tabformer(z_trx, tabf_labels, is_train_step=False)
        self.valid_tabformer_loss(loss_tabformer)

    def training_epoch_end(self, _):
        self.log(f'tabformer/train_tabformer_loss', self.train_tabformer_loss, prog_bar=False)
        # self.train_tabformer_loss reset not required here

    def validation_epoch_end(self, _):
        self.log(f'tabformer/valid_tabformer_loss', self.valid_tabformer_loss, prog_bar=True)
        # self.valid_tabformer_loss reset not required here

    @property
    def seq_encoder(self):
        return TabformerInferenceModule(pretrained_model=self)


class TabformerInferenceModule(torch.nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
        self.model._seq_encoder.is_reduce_sequence = True
        self.model._seq_encoder.add_cls_output = False 
        if self.model.hparams.inference_pooling_strategy=='stat':
            self.stat_pooler = StatPooling()

    def forward(self, batch: PaddedBatch):
        z_trx = self.model.trx_encoder(batch)
        payload = z_trx.payload.view(z_trx.payload.shape[:-1] + (-1, self.model.feature_emb_dim))
        payload = self.model.feature_encoder(payload)
        encoded_trx = PaddedBatch(payload=payload, length=z_trx.seq_lens)
        out = self.model._seq_encoder(encoded_trx)

        if self.model.hparams.inference_pooling_strategy=='stat':
            stats = self.stat_pooler(z_trx)
            out = torch.cat([stats, out], dim=1)  # out: B, 5H
        if self.model.hparams.norm_predict:
            out = out / (out.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5)
        return out
