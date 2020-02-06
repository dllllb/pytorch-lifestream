if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    
import pickle
import logging
import numpy as np
import torch

from dltranz.seq_encoder import RnnEncoder, LastStepEncoder
from dltranz.trx_encoder import TrxEncoder
from dltranz.metric_learn.dataset import create_train_data_loader, create_valid_data_loader
from dltranz.cpc import CPC_Ecoder, run_experiment
from dltranz.util import init_logger, get_conf
from dltranz.data_load import TrxDataset, ConvertingTrxDataset
from metric_learning import prepare_embeddings

logger = logging.getLogger(__name__)

def create_ds(train_data, valid_data, conf):

    train_ds = ConvertingTrxDataset(TrxDataset(train_data))
    valid_ds = ConvertingTrxDataset(TrxDataset(valid_data))

    return train_ds, valid_ds

def main(args=None):
    conf = get_conf(args)

    with open(conf['dataset.train_path'], 'rb') as f:
        data = pickle.load(f)
    data = list(prepare_embeddings(data, conf))

    valid_ix = np.arange(len(data))
    valid_ix = np.random.choice(valid_ix, size=int(len(data) * conf['dataset.valid_size']), replace=False)

    train_data = [rec for i, rec in enumerate(data) if i not in valid_ix]
    valid_data = [rec for i, rec in enumerate(data) if i in valid_ix]

    train_ds, valid_ds = create_ds(train_data, valid_data, conf)

    logger.info(f'Train data len: {len(train_data)}, Valid data len: {len(valid_data)}')

    trx_e = TrxEncoder(conf['params.trx_encoder'])
    rnn_e = RnnEncoder(TrxEncoder.output_size(conf['params.trx_encoder']), conf['params.rnn'])
    cpc_e = CPC_Ecoder(trx_e, rnn_e, conf['params.train.cpc'])

    run_experiment(train_ds, valid_ds, cpc_e, conf)

    l = LastStepEncoder()
    model = torch.nn.Sequential(trx_e, rnn_e, l)

    if conf.get('save_model', False):
        torch.save(model, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')

if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('dataset_preparation')

    main()