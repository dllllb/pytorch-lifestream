import logging
from itertools import islice

import numpy as np
import torch

from tqdm.auto import tqdm
from dltranz.metric_learn.ml_models import ml_model_by_type
from dltranz.seq_encoder import RnnEncoder, LastStepEncoder
from dltranz.trx_encoder import TrxEncoder
from dltranz.metric_learn.dataset import create_train_data_loader, create_valid_data_loader
from dltranz.cpc import CPC_Ecoder, run_experiment
from dltranz.util import init_logger, get_conf, switch_reproducibility_on
from dltranz.data_load import TrxDataset, ConvertingTrxDataset, read_data_gen, SameTimeShuffleDataset, \
    AllTimeShuffleDataset
from metric_learning import prepare_embeddings, shuffle_client_list_reproducible

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    switch_reproducibility_on()


def create_ds(train_data, valid_data, conf):

    train_ds = ConvertingTrxDataset(TrxDataset(train_data))
    valid_ds = ConvertingTrxDataset(TrxDataset(valid_data))

    return train_ds, valid_ds


def target_rm(seq):
    for rec in seq:
        rec['target'] = -1
        yield rec


def main(args=None):
    conf = get_conf(args)

    data = read_data_gen(conf['dataset.train_path'])
    data = tqdm(data)
    if 'max_rows' in conf['dataset']:
        data = islice(data, conf['dataset.max_rows'])
    data = target_rm(data)
    data = prepare_embeddings(data, conf, is_train=True)
    data = shuffle_client_list_reproducible(conf, data)
    data = list(data)

    valid_ix = np.arange(len(data))
    valid_ix = np.random.choice(valid_ix, size=int(len(data) * conf['dataset.valid_size']), replace=False)
    valid_ix = set(valid_ix.tolist())

    train_data = [rec for i, rec in enumerate(data) if i not in valid_ix]
    valid_data = [rec for i, rec in enumerate(data) if i in valid_ix]

    train_ds, valid_ds = create_ds(train_data, valid_data, conf)
    if conf['params.train'].get('same_time_shuffle', False):
        train_ds = SameTimeShuffleDataset(train_ds)
        logger.info('SameTimeShuffle used')
    if conf['params.train'].get('all_time_shuffle', False):
        train_ds = AllTimeShuffleDataset(train_ds)
        logger.info('AllTimeShuffle used')

    logger.info(f'Train data len: {len(train_data)}, Valid data len: {len(valid_data)}')

    cpc_e = ml_model_by_type(conf['params.model_type'])(conf['params'])

    run_experiment(train_ds, valid_ds, cpc_e, conf)

    if conf.get('save_model', False):
        trx_e, rnn_e = cpc_e.trx_encoder, cpc_e.seq_encoder
        l = LastStepEncoder()
        enc_agr_model = torch.nn.Sequential(trx_e, rnn_e, l)

        torch.save(enc_agr_model, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')

if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('dataset_preparation')

    main()
