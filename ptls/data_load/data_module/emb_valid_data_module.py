import json
import logging

import pytorch_lightning as pl
import numpy as np
from embeddings_validation.file_reader import TargetFile, ID_TYPE_MAPPING
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import random
import glob

from ptls.data_load import padded_collate_emb_valid, IterableChain, IterableAugmentations
from ptls.data_load.parquet_dataset import ParquetFiles, ParquetDataset


logger = logging.getLogger(__name__)


class EmbValidDataModule(pl.LightningDataModule):
    def __init__(self, conf, pl_module):
        super().__init__()

        self._type = conf['type']
        assert self._type in ('map', 'iterable')

        self.setup_conf = conf['setup']
        self.train_conf = conf['train']
        self.valid_conf = conf['valid']
        self.test_conf = conf.get('test', self.valid_conf)

        self.col_id = self.setup_conf['col_id']
        self.col_id_dtype = {
            'str': str,
            'int': int,
        }[self.setup_conf['col_id_dtype']]
        self.col_target = self.setup_conf['col_target']
        self.category_names = pl_module.seq_encoder.category_names
        self.category_names.add('event_time')
        self.category_max_size = pl_module.seq_encoder.category_max_size

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self._train_targets = None
        self._valid_targets = None
        self._test_targets = None

        self._fold_info = None
        self.conf = conf

    def prepare_data(self):
        if 'dataset_files' in self.setup_conf:
            self.setup_iterable_files()
        elif 'dataset_parts' in self.setup_conf:
            self.setup_iterable_parts()
        else:
            raise AttributeError(f'Missing dataset definition. One of `dataset_parts` or `dataset_files` expected')

    def setup_iterable_files(self):
       if (self.setup_conf.get('use_files_partially', None)):
           n_train = len(glob.glob(self.setup_conf['dataset_files.train_data_path'] + "/*.parquet"))
           ixes = list(range(n_train))
           train_ixes, test_ixes = train_test_split(ixes, test_size=int(n_train * (1 - self.setup_conf['train_part'])), shuffle=True)
           train_data_files = ParquetFiles(self.setup_conf['dataset_files.train_data_path'], train_ixes).data_files
           if self.setup_conf['same_file_for_test']:
               test_ixes = random.sample(test_ixes, int(n_train * self.setup_conf['test_part']))
               test_data_files = ParquetFiles(self.setup_conf['dataset_files.train_data_path'], test_ixes).data_files
           else:
               n_test = len(glob.glob(self.setup_conf['dataset_files.test_data_path'] + "/*.parquet"))
               test_ixes = random.sample(ixes, int(n_test * self.setup_conf['test_part']))
               test_data_files = ParquetFiles(self.setup_conf['dataset_files.test_data_path'], test_ixes).data_files
       else:
           train_data_files = ParquetFiles(self.setup_conf['dataset_files.train_data_path']).data_files
           test_data_files = ParquetFiles(self.setup_conf['dataset_files.test_data_path']).data_files

       self.train_dataset = ParquetDataset(
           train_data_files,
           shuffle_files=True if self._type == 'iterable' else False
       )
       self.test_dataset = ParquetDataset(
           test_data_files,
           shuffle_files=False
       )
       self.test_dataset = list(tqdm(iter(self.test_dataset)))
       logger.info(f'Loaded {len(self.test_dataset)} for test')


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=padded_collate_emb_valid,
            shuffle=False if self._type == 'iterable' else True,
            num_workers=self.train_conf['num_workers'],
            batch_size=self.train_conf['batch_size'],
            drop_last=self.train_conf.get('drop_last', False)
        )

    def test_dataloader(self):
         return DataLoader(
             dataset=self.test_dataset,
             collate_fn=padded_collate_emb_valid,
             num_workers=self.test_conf['num_workers'],
             batch_size=self.test_conf['batch_size'],
         )
