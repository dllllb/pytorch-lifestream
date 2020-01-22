if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import logging
import pickle

from dltranz.util import init_logger, get_conf
from metric_learning import prepare_embeddings
from dltranz.metric_learn.inference_tools import load_model, score_part_of_data

logger = logging.getLogger(__name__)


def read_dataset(path, conf):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    logger.info(f'loaded {len(data)} records')

    for rec in data:
        rec['target'] = -1

    data = list(prepare_embeddings(data, conf))

    return data


def main(args=None):
    conf = get_conf(args)

    model = load_model(conf)
    columns = conf['output.columns']

    train_data = read_dataset(conf['dataset.train_path'], conf)
    test_data = read_dataset(conf['dataset.test_path'], conf)
    score_part_of_data(None, train_data+test_data, columns, model, conf)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('retail_embeddings_projects.embedding_tools')

    main(None)
