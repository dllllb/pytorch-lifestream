import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F
import torchmetrics

from ptls.lightning_modules.AbsModule import ABSModule
from ptls.seq_encoder.rnn_encoder import RnnSeqEncoder
from ptls.trx_encoder import PaddedBatch


class CPC_Loss(nn.Module):
    def __init__(self, n_negatives):
        super().__init__()
        self.n_negatives = n_negatives

    def _get_preds(self, base_embeddings, mapped_ctx_embeddings):
        batch_size, max_seq_len, emb_size = base_embeddings.payload.shape
        _, _, _, n_forward_steps = mapped_ctx_embeddings.payload.shape
        seq_lens = mapped_ctx_embeddings.seq_lens
        device = mapped_ctx_embeddings.payload.device

        len_mask = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
        len_mask = (len_mask < seq_lens.unsqueeze(1).expand(-1, max_seq_len)).float()

        possible_negatives = base_embeddings.payload.view(batch_size * max_seq_len, emb_size)

        mask = len_mask.unsqueeze(0).expand(batch_size, *len_mask.shape).clone()

        mask = mask.reshape(batch_size, -1)
        sample_ids = torch.multinomial(mask, self.n_negatives)
        neg_samples = possible_negatives[sample_ids]

        positive_preds, neg_preds = [], []
        len_mask_exp = len_mask.unsqueeze(-1).unsqueeze(-1).to(device).expand(-1, -1, emb_size, n_forward_steps)
        trimmed_mce = mapped_ctx_embeddings.payload.mul(len_mask_exp)  # zero context vectors by sequence lengths
        for i in range(1, n_forward_steps + 1):
            ce_i = trimmed_mce[:, 0:max_seq_len - i, :, i - 1]
            be_i = base_embeddings.payload[:, i:max_seq_len]

            positive_pred_i = ce_i.mul(be_i).sum(axis=-1)
            positive_preds.append(positive_pred_i)

            neg_pred_i = ce_i.matmul(neg_samples.transpose(-2, -1))
            neg_pred_i = neg_pred_i
            neg_preds.append(neg_pred_i)

        return positive_preds, neg_preds

    def forward(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings
        device = mapped_ctx_embeddings.payload.device
        positive_preds, neg_preds = self._get_preds(base_embeddings, mapped_ctx_embeddings)

        step_losses = []
        for positive_pred_i, neg_pred_i in zip(positive_preds, neg_preds):
            step_loss = -F.log_softmax(torch.cat([positive_pred_i.unsqueeze(-1), neg_pred_i], dim=-1), dim=-1)[:, :,
                         0].mean()
            step_losses.append(step_loss)

        loss = torch.stack(step_losses).mean()
        return loss

    def cpc_accuracy(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings
        positive_preds, neg_preds = self._get_preds(base_embeddings, mapped_ctx_embeddings)

        batch_size, max_seq_len, emb_size = base_embeddings.payload.shape
        seq_lens = mapped_ctx_embeddings.seq_lens
        device = mapped_ctx_embeddings.payload.device

        len_mask = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
        len_mask = (len_mask < seq_lens.unsqueeze(1).expand(-1, max_seq_len)).float()

        total, accurate = 0, 0
        for i, (positive_pred_i, neg_pred_i) in enumerate(zip(positive_preds, neg_preds)):
            i_mask = len_mask[:, (i + 1):max_seq_len].to(device)
            total += i_mask.sum().item()
            accurate += (((positive_pred_i.unsqueeze(-1).expand(*neg_pred_i.shape) > neg_pred_i) \
                          .sum(dim=-1) == self.n_negatives) * i_mask).sum().item()
        return accurate / total


class CpcAccuracyPL(torchmetrics.Metric):
    def __init__(self, loss):
        super().__init__(compute_on_step=False)

        self.loss = loss
        self.add_state('batch_results', default=torch.tensor([0.0]))
        self.add_state('total_count', default=torch.tensor([0]))

    def update(self, *input):
        self.batch_results += self.loss.cpc_accuracy(*input)
        self.total_count += 1

    def compute(self):
        return self.batch_results / self.total_count.float()


class CpcModule(ABSModule):
    def __init__(self, params, seq_encoder=None):
        if seq_encoder is not None and not isinstance(seq_encoder, RnnSeqEncoder):
            raise NotImplementedError(f'Only rnn encoder supported in CpcModule. Found {type(seq_encoder)}')

        if params['encoder_type'] != 'rnn':
            raise NotImplementedError(f'Only rnn encoder supported in CpcModule. Found {params.encoder_type}')

        super().__init__(params, seq_encoder)

        linear_size = self.seq_encoder.model[0].output_size
        embedding_size = self.seq_encoder.embedding_size
        self._linears = torch.nn.ModuleList([torch.nn.Linear(embedding_size, linear_size)
                                             for _ in range(params['cpc.n_forward_steps'])])

    @property
    def metric_name(self):
        return 'cpc_accuracy'

    @property
    def is_requires_reduced_sequence(self):
        return False

    def get_loss(self):
        return CPC_Loss(n_negatives=self.hparams.params['cpc.n_negatives'])

    def get_validation_metric(self):
        return CpcAccuracyPL(self._loss)

    def shared_step(self, x, y):
        trx_encoder = self._seq_encoder.model[0]
        seq_encoder = self._seq_encoder.model[1]

        base_embeddings = trx_encoder(x)
        context_embeddings = seq_encoder(base_embeddings)

        me = []
        for l in self._linears:
            me.append(l(context_embeddings.payload))
        mapped_ctx_embeddings = PaddedBatch(torch.stack(me, dim=3), context_embeddings.seq_lens)

        return (base_embeddings, context_embeddings, mapped_ctx_embeddings), y
