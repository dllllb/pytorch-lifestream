import pickle
import logging
import numpy as np

from dltranz.seq_encoder import RnnEncoder
from dltranz.trx_encoder import TrxEncoder
from dltranz.metric_learn.dataset import create_train_data_loader, create_valid_data_loader
from dltranz.cpc import CPC_Ecoder, run_experiment
from dltranz.util import init_logger, get_conf

from metric_learning import prepare_embeddings

logger = logging.getLogger(__name__)

def main(args=None):
    conf = get_conf(args)

    with open(conf['dataset.path'], 'rb') as f:
        data = pickle.load(f)
    data = list(prepare_embeddings(data, conf))

    valid_ix = np.arange(len(data))
    valid_ix = np.random.choice(valid_ix, size=int(len(data) * conf['dataset.valid_size']), replace=False)

    train_data = [rec for i, rec in enumerate(data) if i not in valid_ix]
    valid_data = [rec for i, rec in enumerate(data) if i in valid_ix]

    logger.info(f'Train data len: {len(train_data)}, Valid data len: {len(valid_data)}')

    trx_e = TrxEncoder(conf['trx_encoder'])
    rnn_e = RnnEncoder(8, conf['rnn'])
    cpc_e = CPC_Ecoder(trx_e, rnn_e)

    return run_experiment(train_data, valid_data, cpc_e, conf)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('dataset_preparation')

    main()