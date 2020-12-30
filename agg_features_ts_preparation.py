import logging
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from dltranz.seq_encoder.agg_feature_model import AggFeatureModel
from dltranz.data_load import ConvertingTrxDataset
from dltranz.metric_learn.dataset import SplittingDataset, split_strategy, TargetEnumeratorDataset, \
    collate_splitted_rows
from dltranz.train import score_model
from dltranz.util import init_logger, get_conf, switch_reproducibility_on
from ml_inference import read_dataset

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    switch_reproducibility_on()


def main(args=None):
    conf = get_conf(sys.argv[2:])

    model = AggFeatureModel(conf['params.trx_encoder'])

    data = read_dataset(conf['dataset.train_path'], conf)
    last_date = math.floor(max([x['event_time'][-1] for x in data]))
    first_date = math.floor(min([x['event_time'][0] for x in data]))
    first_date = first_date + (last_date - first_date) // 2

    chunk_size = conf['params.chunk_size']
    n_chunks, remainder = divmod(len(data), chunk_size)
    n_chunks += (remainder > 0)

    if conf.get('save_model', False):
        save_dir = os.path.dirname(conf['model_path.model'])
        os.makedirs(save_dir, exist_ok=True)

        torch.save(model, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')

    os.makedirs(conf['dataset_output.path'], exist_ok=True)
    p = 0
    for i in range(n_chunks):
        chunk = data[p:p+chunk_size]
        dataset = SplittingDataset(
            chunk,
            split_strategy.CutByDays(first_date, last_date)
        )
        dataset = TargetEnumeratorDataset(dataset)
        dataset = ConvertingTrxDataset(dataset)

        loader = DataLoader(
            dataset=dataset,
            shuffle=False,
            collate_fn=collate_splitted_rows,
            num_workers=conf['params.inference'].get('num_workers', 0),
            batch_size=conf['params.inference.batch_size'],
        )

        _, pred = score_model(model, loader, conf['params'])
        pred = pred.reshape(len(dataset), len(pred) // len(dataset), -1)
        pred = np.swapaxes(pred, 0, 1)
        pred = torch.from_numpy(pred)

        torch.save(pred, os.path.join(conf['dataset_output.path'], f'{i:03d}.pt'))

        p += chunk_size
        logger.info(f'Chunks finished: {i+1} ')


def load_agg_data(conf):
    dataset_path = conf['dataset.path']
    data = []
    for i in range(4):
        chunk = torch.load(os.path.join(dataset_path, f'{i:03d}.pt')).transpose(0, 1)
        data.append(chunk)

    data = torch.cat(data, dim=0)
    return data


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')

    main()
