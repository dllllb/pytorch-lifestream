import logging
import torch
import torch.nn as nn
import numpy as np

from ignite.metrics import Loss, RunningAverage
from torch.autograd import Variable
from torch.nn import functional as F

from dltranz.trx_encoder import PaddedBatch
from dltranz.experiment import update_model_stats, CustomMetric
from dltranz.metric_learn.metric import BatchRecallTop
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model
from dltranz.data_load import create_train_loader, create_validation_loader

logger = logging.getLogger(__name__)

class CPC_Ecoder(nn.Module):
    def __init__(self, trx_encoder, seq_encoder, conf):
        super().__init__()
        self.trx_encoder = trx_encoder
        self.seq_encoder = seq_encoder
        embedding_size = conf['embedding_size']
        linear_size = conf['linear_size']
        self.linears = nn.ModuleList([nn.Linear(embedding_size, linear_size) for i in range(conf['n_forward_steps'])])

    def forward(self, x: PaddedBatch):
        base_embeddings = self.trx_encoder(x)
        context_embeddings = self.seq_encoder(base_embeddings)

        me = []
        for l in self.linears:
            me.append(l(context_embeddings.payload))
        mapped_ctx_embeddings = PaddedBatch(torch.stack(me, dim=3), context_embeddings.seq_lens)

        return base_embeddings, context_embeddings, mapped_ctx_embeddings

class CPC_Loss(nn.Module):
    def __init__(self, n_negatives):
        super().__init__()
        self.n_negatives = n_negatives

    def _get_preds(self, base_embeddings, mapped_ctx_embeddings):
        batch_size, max_seq_len, emb_size = base_embeddings.payload.shape
        _, _, _, n_forward_steps = mapped_ctx_embeddings.payload.shape
        seq_lens = mapped_ctx_embeddings.seq_lens
        device=mapped_ctx_embeddings.payload.device

        len_mask = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size,-1)
        len_mask = (len_mask<seq_lens.unsqueeze(1).expand(-1,max_seq_len)).float()

        possible_negatives = base_embeddings.payload.view(batch_size*max_seq_len, emb_size)

        mask = len_mask.unsqueeze(0).expand(batch_size, *len_mask.shape).clone()
        for i in range(batch_size):
            mask[i,i] = 0 # ignoring current sample embeddings

        mask = mask.reshape(batch_size, -1)
        sample_ids = torch.multinomial(mask, self.n_negatives)
        neg_samples = possible_negatives[sample_ids]

        positive_preds, neg_preds = [], []
        len_mask_exp = len_mask.unsqueeze(-1).unsqueeze(-1).to(device).expand(-1, -1, emb_size, n_forward_steps)
        trimmed_mce = mapped_ctx_embeddings.payload.mul(len_mask_exp)  # zero context vectors by sequence lengths
        for i in range(1, n_forward_steps+1):
            ce_i = trimmed_mce[:,0:max_seq_len-i, :, i-1]
            be_i = base_embeddings.payload[:,i:max_seq_len]

            positive_pred_i = ce_i.mul(be_i).sum(axis=-1)
            positive_preds.append(positive_pred_i)

            neg_pred_i = ce_i.matmul(neg_samples.transpose(-2, -1))
            neg_pred_i = neg_pred_i
            neg_preds.append(neg_pred_i)

        return positive_preds, neg_preds

    def forward(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings
        device=mapped_ctx_embeddings.payload.device
        positive_preds, neg_preds = self._get_preds(base_embeddings, mapped_ctx_embeddings)

        step_losses = []
        for positive_pred_i, neg_pred_i in zip(positive_preds, neg_preds):
            step_loss = -F.log_softmax(torch.cat([positive_pred_i.unsqueeze(-1), neg_pred_i], dim=-1), dim=-1)[:,:,0].mean()
            step_losses.append(step_loss)

        loss = torch.stack(step_losses).mean()
        return loss

    def cpc_accuracy(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings
        positive_preds, neg_preds = self._get_preds(base_embeddings, mapped_ctx_embeddings)

        batch_size, max_seq_len, emb_size = base_embeddings.payload.shape
        seq_lens = mapped_ctx_embeddings.seq_lens
        device=mapped_ctx_embeddings.payload.device

        len_mask = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size,-1)
        len_mask = (len_mask<seq_lens.unsqueeze(1).expand(-1,max_seq_len)).float()

        total, accurate = 0, 0
        for i, (positive_pred_i, neg_pred_i) in enumerate(zip(positive_preds, neg_preds)):
            i_mask = len_mask[:,(i+1):max_seq_len].to(device)
            total += i_mask.sum().item()
            accurate += (((positive_pred_i.unsqueeze(-1).expand(*neg_pred_i.shape) > neg_pred_i) \
                .sum(dim=-1) == self.n_negatives)*i_mask).sum().item()

        return accurate / total


def run_experiment(train_ds, valid_ds, model, conf):
    import time
    start = time.time()

    params = conf['params']

    loss = CPC_Loss(n_negatives=params['train.cpc.n_negatives'])

    valid_metric = {
        'loss': RunningAverage(Loss(loss)),
        'cpc accuracy': CustomMetric(lambda outputs, y: loss.cpc_accuracy(outputs, y))
    }

    optimizer = get_optimizer(model, params)
    scheduler = get_lr_scheduler(optimizer, params)

    train_loader = create_train_loader(train_ds, params['train'])
    valid_loader = create_validation_loader(valid_ds, params['valid'])

    metric_values = fit_model(
        model,
        train_loader,
        valid_loader,
        loss,
        optimizer,
        scheduler,
        params,
        valid_metric,
        train_handlers=[])

    exec_sec = time.time() - start

    if conf.get('save_model', False):
        torch.save(model, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')

    results = {
        'exec-sec': exec_sec,
        'metrics': metric_values,
    }

    stats_file = conf.get('stats.path', None)

    if stats_file is not None:
        update_model_stats(stats_file, params, results)
    else:
        return results
