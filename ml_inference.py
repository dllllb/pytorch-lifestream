import logging
from itertools import islice

import numpy as np
import torch
from tqdm.auto import tqdm

from dltranz.data_load import read_data_gen
from dltranz.util import init_logger, get_conf, switch_reproducibility_on
from metric_learning import prepare_embeddings
from dltranz.metric_learn.inference_tools import score_part_of_data
from dltranz.metric_learn.ml_models import load_encoder_for_inference

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    switch_reproducibility_on()


def fill_target(seq):
    for rec in seq:
        rec['target'] = -1
        yield rec


def read_dataset(path, conf):
    data = read_data_gen(path)
    data = tqdm(data)
    if 'max_rows' in conf['dataset']:
        data = islice(data, conf['dataset.max_rows'])
    data = fill_target(data)
    data = prepare_embeddings(data, conf, is_train=False)
    data = list(data)

    logger.info(f'loaded {len(data)} records')
    return data


def main(args=None):
    conf = get_conf(args)
    model = load_encoder_for_inference(conf)
    columns = conf['output.columns']

    train_data = read_dataset(conf['dataset.train_path'], conf)
    if conf['dataset'].get('test_path', None) is not None:
        test_data = read_dataset(conf['dataset.test_path'], conf)
    else:
        test_data = []
    score_part_of_data(None, train_data+test_data, columns, model, conf)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('retail_embeddings_projects.embedding_tools')

    main(None)
