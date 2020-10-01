import logging
from itertools import islice

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from dltranz.data_load import read_data_gen, ConvertingTrxDataset
from dltranz.metric_learn.dataset import split_strategy, \
    collate_splitted_rows
from dltranz.train import score_model
from dltranz.util import init_logger, get_conf, switch_reproducibility_on
from metric_learning import prepare_embeddings
from dltranz.metric_learn.inference_tools import infer_part_of_data, save_scores
from dltranz.metric_learn.ml_models import load_encoder_for_inference

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    switch_reproducibility_on()


def fill_target(seq, conf):
    col_id = conf['output.columns'][0]
    for rec in seq:
        rec['target'] = int(rec[col_id])
        yield rec


def read_dataset(path, conf):
    data = read_data_gen(path)
    if 'max_rows' in conf['dataset']:
        data = islice(data, conf['dataset.max_rows'])
    data = fill_target(data, conf)
    data = prepare_embeddings(data, conf, is_train=False)
    data = list(data)

    logger.info(f'loaded {len(data)} records')
    return data


class SplittingDataset(Dataset):
    def __init__(self, base_dataset, splitter):
        self.base_dataset = base_dataset
        self.splitter = splitter

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]

        feature_arrays = row['feature_arrays']
        local_date = row['event_time']

        indexes = self.splitter.split(local_date)
        data = [{k: v[ix] for k, v in feature_arrays.items()} for ix in indexes]
        data = [(x, row['target']) for x in data]
        return data


def infer_part_of_data(part_num, part_data, columns, model, conf, lock_obj=None):
    if lock_obj:
        lock_obj.acquire()

    if part_num is None:
        logger.info(f'Start to score data ({len(part_data)} records)')
    else:
        logger.info(f'Start to score {part_num} part of data ({len(part_data)} records)')

    if conf['dataset.preprocessing.add_seq_len'] and 'seq_len' not in columns:
        columns.append('seq_len')  # change list object

    valid_ds = SplittingDataset(
        part_data,
        split_strategy.create(**conf['params.valid.split_strategy'])
    )
    valid_ds = ConvertingTrxDataset(valid_ds)
    valid_loader = DataLoader(
        dataset=valid_ds,
        shuffle=False,
        collate_fn=collate_splitted_rows,
        num_workers=conf['params.valid'].get('num_workers', 0),
        batch_size=conf['params.valid.batch_size'],
    )
    target_labels, pred = score_model(model, valid_loader, conf['params'])

    if conf['params.device'] != 'cpu':
        torch.cuda.empty_cache()
        logger.info('torch.cuda.empty_cache()')
    if lock_obj:
        lock_obj.release()

    if len(pred.shape) == 1:
        pred = pred.reshape(-1, 1)

    df_scores_cols = [f'v{i:003d}' for i in range(pred.shape[1])]
    df_scores = pd.DataFrame(pred, columns=df_scores_cols)

    col_id = conf['output.columns'][0]
    df_scores[col_id] = target_labels

    df_scores = df_scores.reindex(columns=[col_id] + df_scores_cols)
    logger.info(f'df_scores examples: {df_scores.shape}:')
    return df_scores


def score_part_of_data(part_num, part_data, columns, model, conf, lock_obj=None):
    df_scores = infer_part_of_data(part_num, part_data, columns, model, conf, lock_obj=lock_obj)

    save_scores(df_scores, part_num, conf['output'])


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
