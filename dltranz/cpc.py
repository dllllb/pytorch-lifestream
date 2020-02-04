import logging
import torch
import torch.nn as nn
import numpy as np

from ignite.metrics import Loss, RunningAverage

from dltranz.trx_encoder import PaddedBatch
from dltranz.experiment import update_model_stats
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

    def forward(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings

        batch_size, max_seq_len, emb_size = base_embeddings.payload.shape
        _, _, _, n_forward_steps = mapped_ctx_embeddings.payload.shape
        seq_lens = mapped_ctx_embeddings.seq_lens

        len_mask = torch.ones(batch_size, max_seq_len)
        for i, l in enumerate(seq_lens):
            len_mask[i, l:] = 0

        possible_negatives = base_embeddings.payload.view(batch_size*max_seq_len, emb_size)

        neg_samples = []
        for i in range(batch_size):
            mask = len_mask.clone()
            mask[i] = 0  # ignoring current sample embeddings
            sample_ids = torch.multinomial(mask.flatten().float(), self.n_negatives)
            neg_samples_i = possible_negatives[sample_ids]
            neg_samples.append(neg_samples_i)

        neg_samples = torch.stack(neg_samples, dim=0)

        step_losses = []
        len_mask_exp = len_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, emb_size, n_forward_steps)
        trimmed_mce = mapped_ctx_embeddings.payload.mul(len_mask_exp)  # zero context vectors by sequence lengths
        for i in range(1, n_forward_steps+1):
            ce_i = trimmed_mce[:,0:max_seq_len-i, :, i-1]
            be_i = base_embeddings.payload[:,i:max_seq_len]
            len_mask_i = len_mask[:,0:max_seq_len-i]
            positive_pred_i = ce_i.mul(be_i).sum(axis=-1).exp()

            neg_pred_i = ce_i.matmul(neg_samples.transpose(-2, -1))
            neg_pred_i = neg_pred_i.sum(axis=-1).exp()

            step_loss = -positive_pred_i.div(neg_pred_i).log().mean()
            step_losses.append(step_loss)

        loss = torch.stack(step_losses).mean()

        return loss

def run_experiment(train_ds, valid_ds, model, conf):
    import time
    start = time.time()

    params = conf['params']

    loss = CPC_Loss(n_negatives=params['train.cpc.n_negatives'])

    valid_metric = {'loss': RunningAverage(Loss(loss))}

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
