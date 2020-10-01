import logging
import os
from functools import partial
from itertools import islice

import numpy as np
import torch
from tqdm.auto import tqdm

from dltranz.data_load import read_data_gen
from dltranz.data_load.lazy_dataset import LazyDataset, DataFiles
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


def swap_target(seq, conf):
    col_id = conf['dataset.col_id']

    for rec in seq:
        rec['target'] = rec[col_id]
        yield rec


def read_file(path, conf):
    data = read_data_gen(path)
    data = swap_target(data, conf)
    data = prepare_embeddings(data, conf, is_train=False)
    return data


def read_dataset(path, conf):
    path, file_name = os.path.split(path)
    if file_name == 'part-*.parquet':
        path = os.path.join(path, file_name)
    elif os.path.splitext(file_name)[1] == '.parquet':
        path = os.path.join(path, file_name, 'part-*.parquet')
    else:
        raise AssertionError(f'Unknown parquet file path format "{path}", "{file_name}"')

    df = DataFiles(path_wc=path, valid_size=0)
    ds = LazyDataset(df.train, partial(read_file, conf=conf))
    return ds


def main(args=None):
    conf = get_conf(args)
    model = load_encoder_for_inference(conf)
    columns = conf['output.columns']

    train_data = read_dataset(conf['dataset.train_path'], conf)
    if conf['dataset'].get('test_path', None) is not None:
        test_data = read_dataset(conf['dataset.test_path'], conf)
        train_data = torch.utils.data.ChainDataset([train_data, test_data])

    score_part_of_data(None, train_data, columns, model, conf)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('retail_embeddings_projects.embedding_tools')

    main(None)
