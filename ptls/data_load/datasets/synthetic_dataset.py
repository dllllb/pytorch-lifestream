import logging
import warnings
from typing import List
import os
from collections import defaultdict
import pandas as pd
from random import shuffle
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

import numpy as np
import torch

from ptls.data_load import IterableChain
from .synthetic_client import SyntheticClient, SimpleSchedule
from ptls.frames import PtlsDataModule
from ptls.frames.supervised import SeqToTargetDataset

logger = logging.getLogger(__name__)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self,
                 clients,
                 seq_len=512,
                 post_processing=None,
                 i_filters: List = None):
        self.clients = clients
        self.seq_len = seq_len
        if i_filters is not None:
            self.post_processing = IterableChain(*i_filters)
        else:
            self.post_processing = post_processing
        if post_processing is not None:
            warnings.warn('`post_processing` parameter is deprecated, use `i_filters`')

    def __getitem__(self, ind):
        item = self.clients[ind].gen_seq(self.seq_len)
        if self.post_processing is not None:
            item = self.post_processing(item)
        return item

    def __len__(self):
        return len(self.clients)

    @staticmethod
    def to_torch(x):
        if type(x) is np.ndarray and x.dtype.kind in ('i', 'f'):
            return torch.from_numpy(x)
        return x


class SyntheticDatasetWriter:
    def __init__(self, config, path, seq_len,
                 n_train_files, n_eval_files, n_test_files,
                 train_per_file, eval_per_file, test_per_file,
                 save_config_name=None, save=False, load=False, n_procs=16):
        self.config = config
        self.schedule = SimpleSchedule(config)
        self.path = path
        self.seq_len = seq_len
        self.n_train_files, self.n_eval_files, self.n_test_files = n_train_files, n_eval_files, n_test_files
        self.train_per_file, self.eval_per_file, self.test_per_file = train_per_file, eval_per_file, test_per_file
        self.n_procs = n_procs

        if save_config_name is not None and save:
            self.config.save_assigners(str(save_config_name))
        if save_config_name is not None and load:
            self.config.load_assigners(str(save_config_name))

    def get_clients(self, n):
        raise NotImplementedError

    def get_datamodule(self, n):
        clients = self.get_clients(n)
        dataset = SyntheticDataset(clients, seq_len=self.seq_len)
        sup_data = PtlsDataModule(
            train_data=SeqToTargetDataset(dataset, target_col_name='class_label', target_dtype=torch.long),
            train_batch_size=256,
            train_num_workers=self.n_procs,
        )
        return sup_data

    def write_dataset(self):
        for mode, n_files, n_per_file in zip(["train", "eval", "test"],
                                             [self.n_train_files, self.n_eval_files, self.n_test_files],
                                             [self.train_per_file, self.eval_per_file, self.test_per_file]):
            for fn in tqdm(range(n_files)):
                folder = os.path.join(self.path, mode)
                os.makedirs(folder, exist_ok=True)
                data = self.get_datamodule(n_per_file)
                df = defaultdict(list)
                for i, batch in enumerate(data.train_dataloader()):
                    x, y = batch
                    x_d = x.payload
                    for k in x_d:
                        df[k].extend(x_d[k].int().tolist())
                    df['class_label'].extend(y.int().tolist())
                df = pd.DataFrame(df)
                df.to_parquet(os.path.join(folder, mode + "_" + str(fn) + ".parquet"))


class MonoTargetSyntheticDatasetWriter(SyntheticDatasetWriter):
    '''
    def get_clients(self, n):
        per_class_n = int(n / 2)
        pool = mp.Pool(self.n_procs)
        sc = partial(SyntheticClient, config=self.config, schedule=self.schedule)
        clients = pool.map(sc, [{0: 0} for _ in range(per_class_n)]+[{0: 1} for _ in range(per_class_n)])
        pool.close()
        pool.join()
        shuffle(clients)
        return clients
    '''
    def get_clients(self, n):
        per_class_n = int(n / 2)
        clients = [SyntheticClient({0: 0}, self.config, self.schedule) for _ in range(per_class_n)] + \
                  [SyntheticClient({0: 1}, self.config, self.schedule) for _ in range(per_class_n)]
        shuffle(clients)
        return clients
