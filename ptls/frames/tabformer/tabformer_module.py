import pytorch_lightning as pl
import torch
import warnings
from torchmetrics import MeanMetric
from typing import Tuple, Dict
from transformers import BertModel, BertConfig

from ptls.frames.bert.losses.query_soft_max import QuerySoftmaxLoss
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn import PBL2Norm
from ptls.data_load.padded_batch import PaddedBatch


class TabformerPretrainModule(pl.LightningModule):
    """Tabformer Model (MLM) from [Tabular Transformers for Modeling Multivariate Time Series](https://arxiv.org/abs/2011.01843)

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
                 feature_encoder: torch.nn.Module,
                 total_steps: int,
                 seq_encoder: AbsSeqEncoder = None,
                 hidden_size: int = None,
                 loss_temperature: float = 20.0,
                 max_lr: float = 0.001,
                 weight_decay: float = 0.0,
                 pct_start: float = 0.1,
                 norm_predict: bool = True,
                 replace_proba: float = 0.1,
                 neg_count: int = 1,
                 log_logits: bool = False,
                 mask_prob=0.15
                 ):

        super().__init__()
        self.save_hyperparameters(ignore=['trx_encoder', 'seq_encoder'])

        self.trx_encoder = trx_encoder
        self.feature_encoder = feature_encoder

        assert not self.trx_encoder.scalers, '`numeric_values` parameter of `trx_encoder` should be == {}. Discretize all numerical features into categorical to use Tabformer model!'
        noisy_embeds = list(self.trx_encoder.embeddings.values())
        assert noisy_embeds, '`embeddings` parameter for `trx_encoder` should contain at least 1 feature!'
        self.feature_emb_dim = noisy_embeds[0].embedding_dim
        self.num_f = len(self.trx_encoder.embeddings) + 1
        assert all(n_emb.embedding_dim == self.feature_emb_dim for n_emb in noisy_embeds), 'Out dimensions for all features in `embeddings` parameter of `trx_encoder` should be equal for Tabformer model!'

        warnings.warn("With Tabformer model set `in` value in `embeddings`parameter of `trx_encoder` equal to actual number of unique feature values + 1")

        if seq_encoder:
            self.seq_encoder = seq_encoder
            self.seq_encoder.is_reduce_sequence = False
        else:
            bert_config = BertConfig(hidden_size=self.feature_emb_dim * self.num_f,
                                     num_attention_heads=self.num_f,
                                     pad_token_id=0)
            self.seq_encoder = BertModel(bert_config, add_pooling_layer=False)

        if self.hparams.norm_predict:
            self.fn_norm_predict = PBL2Norm()

        if hidden_size is None:
            hidden_size = trx_encoder.output_size

        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, self.feature_emb_dim), requires_grad=True)
        self.token_sep = torch.nn.Parameter(torch.randn(1, 1, self.feature_emb_dim), requires_grad=True)

        self.loss_fn = QuerySoftmaxLoss(temperature=loss_temperature, reduce=False)

        self.train_tabformer_loss = MeanMetric()
        self.valid_tabformer_loss = MeanMetric()
        self.mask_prob = mask_prob

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
        outputs = self.seq_encoder(
            inputs_embeds=z.payload,
        )

        print(self.hparams.norm_predict)
        7777777/0
        if self.hparams.norm_predict:
            out = self.fn_norm_predict(out)
        return out

    def get_masks_and_labels(self, batch: PaddedBatch) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
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

        return *self.tabformer_mask(feature_tensors, probability_matrix), random_words

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

    def loss_tabformer(self, x: PaddedBatch, is_train_step):
        out = self.forward(x)

        target = x.payload[mask].unsqueeze(1)  # N, 1, H
        predict = out[mask].unsqueeze(1)  # N, 1, H
        neg_ix = self.get_neg_ix(mask)
        negative = out[neg_ix[0], neg_ix[1]]  # N, nneg, H
        loss = self.loss_fn(target, predict, negative)

        if is_train_step and self.hparams.log_logits:
            with torch.no_grad():
                logits = self.loss_fn.get_logits(target, predict, negative)
            self.logger.experiment.add_histogram('tabformer/logits',
                                                 logits.flatten().detach(), self.global_step)
        return loss

    def training_step(self, batch, batch_idx):
        tabf_labels, MASK_token_mask, RANDOM_token_mask, random_words = self.get_masks_and_labels(batch)
        z_trx = self.trx_encoder(batch)  # PB: B, T, H

        payload = z_trx.payload.view(z_trx.payload.shape[:-1] + (-1, self.feature_emb_dim))
        payload[MASK_token_mask] = self.token_mask
        payload[RANDOM_token_mask] = random_words[RANDOM_token_mask]
        payload = self.feature_encoder(payload)

        z_trx._payload = payload
        z_trx.tabf_labels = tabf_labels

        loss_tabformer = self.loss_tabformer(z_trx, is_train_step=True)
        self.train_tabformer_loss(loss_tabformer)
        loss_tabformer = loss_tabformer.mean()
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
        z_trx.tabf_labels = tabf_labels


        loss_tabformer = self.loss_tabformer(z_trx, is_train_step=False)
        self.valid_tabformer_loss(loss_tabformer)

    def training_epoch_end(self, _):
        self.log(f'tabformer/train_tabformer_loss', self.train_tabformer_loss, prog_bar=False)
        # self.train_tabformer_loss reset not required here

    def validation_epoch_end(self, _):
        self.log(f'tabformer/valid_tabformer_loss', self.valid_tabformer_loss, prog_bar=True)
        # self.valid_tabformer_loss reset not required here

