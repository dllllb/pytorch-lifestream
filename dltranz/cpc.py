import logging
import torch.nn as nn
import numpy as np

from trx_encoder import PaddedBatch
from experiment import update_model_stats
from dltranz.metric_learn.metric import BatchRecallTop
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model
from dltranz.data_load import create_train_loader, create_validation_loader

logger = logging.getLogger(__name__)

class CPC_Ecoder(nn.Module):
    def __init__(self, trx_encoder, seq_encoder):
        self.trx_encoder = trx_encoder
        self.seq_encoder = seq_encoder

    def forward(self, x: PaddedBatch):
        z = self.trx_encoder(x)
        c = self.seq_encoder(z)

        return z, c

class CPC_Loss(nn.Module):
    def __init__(self, n_forward_steps, linear_size):
        super().__init__()
        self.forward_steps = n_forward_steps
        self.linear = nn.Linear(linear_size, n_forward_steps)

    def forward(self, embeddings):
        base_embeddings, context_embeddings = embeddings
        mapped_embeddings = self.linear(context_embeddings)

        batch_size, max_seq_len, emb_size = context_embeddings.shape

        positive_pairs = []
        negative_pairs = []
        for i in range(self.n_forward_steps):
            ce_i = mapped_embeddings[:,0:max_seq_len-i-1,:]
            be_i = base_embeddings[:,i+1:max_seq_len,:]

        positive_pairs = np.array(positive_pairs)
        negative_pairs = np.array(negative_pairs)

        positive_pred = (positive_pairs[:, 0]*positive_pairs[:, 1]).sum(axis=1).exp()
        negative_pred = (negative_pairs[:, :, 0]*negative_pairs[:, :, 1]).sum(axis=1).exp().sum(axis=1)
        loss = -(positive_pred/negative_pred).log().mean()

        return loss

def run_experiment(train_ds, valid_ds, model, conf):
    import time
    start = time.time()

    stats_file = conf['stats.path']
    params = conf['params']

    loss = CPC_Loss(conf['cpc.n_forward_steps'], conf['cpc.linear_size'])

    valid_metric = {'BatchRecallTop': BatchRecallTop(k=params['valid.split_strategy.split_count'] - 1)}
    optimizer = get_optimizer(model, params)
    scheduler = get_lr_scheduler(optimizer, params)

    train_loader = create_train_loader(train_ds, config['train'])
    valid_loader = create_validation_loader(valid_ds, config['valid'])

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
        'Recall_top_K': metric_values,
    }

    if conf.get('log_results', True):
        update_model_stats(stats_file, params, results)
