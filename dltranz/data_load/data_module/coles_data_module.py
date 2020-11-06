"""
Creates data loaders for contrastive learning
Each record pass throw the sampler and splits into views of the main record
X - is feature_arrays, sub-samples of main event sequences
Y - is Id of main record, for positive item identification

    Input:        -> Output
dict of arrays    ->  X,                 Y
client_1          -> client_1_smpl_1, client_1_id
                     client_1_smpl_2, client_1_id
                     client_1_smpl_3, client_1_id
client_2          -> client_2_smpl_1, client_2_id
                     client_2_smpl_2, client_2_id
                     client_2_smpl_3, client_2_id
"""
import logging

import pytorch_lightning as pl
from glob import glob
from torch.utils.data import DataLoader
import os

from dltranz.data_load import padded_collate, IterableAugmentations, IterableChain
from dltranz.data_load.augmentations.dropout_trx import DropoutTrx
from dltranz.data_load.iterable_processing.feature_filter import FeatureFilter
from dltranz.data_load.iterable_processing.iterable_shuffle import IterableShuffle
from dltranz.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from dltranz.data_load.iterable_processing.target_extractor import TargetExtractor
from dltranz.data_load.list_splitter import ListSplitter
from dltranz.data_load.parquet_dataset import ParquetDataset
from dltranz.data_load.partitioned_dataset import PartitionedDataset, PartitionedDataFiles
from dltranz.metric_learn.dataset import split_strategy
from dltranz.metric_learn.dataset.splitting_dataset import IterableSplittingDataset

logger = logging.getLogger(__name__)


class ColesDataModuleIterableTrain(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.setup_conf = conf['setup']
        self.train_conf = conf['train']
        self.valid_conf = conf['valid']

        self.col_id = self.setup_conf['col_id']

        self.train_dataset = None
        self.valid_dataset = None

    def setup(self, stage=None):
        if 'dataset_parts' in self.setup_conf:
            self.setup_parts()
        elif 'dataset_files' in self.setup_conf:
            self.setup_files()
        else:
            raise AttributeError(f'Missing dataset definition. One of `dataset_parts` or `dataset_files` expected')

    def setup_parts(self):
        data_files = PartitionedDataFiles(**self.setup_conf['dataset_parts'])

        if self.setup_conf['split_by'] == 'hash_id':
            splitter = ListSplitter(
                data_files.hash_parts,
                valid_size=self.setup_conf['valid_size'],
                seed=self.setup_conf['valid_split_seed'],
            )
            logger.info(f'Prepared splits: '
                        f'{len(data_files.dt_parts)} parts in dt_train, {len(data_files.dt_parts)} parts in dt_valid, '
                        f'{len(splitter.train)} parts in hash_train, {len(splitter.valid)} parts in hash_valid')

            self.train_dataset = PartitionedDataset(
                data_files.data_path, data_files.dt_parts, splitter.train, col_id=self.col_id,
                post_processing=self.train_post_processing,
                shuffle_files=True,
            )
            self.valid_dataset = PartitionedDataset(
                data_files.data_path, data_files.dt_parts, splitter.valid, col_id=self.col_id,
                post_processing=self.valid_post_processing,
                shuffle_files=False,
            )
        else:
            raise AttributeError(f'Unknown split strategy: {self.setup_conf.split_by}')

    def setup_files(self):
        if self.setup_conf['split_by'] == 'files':
            file_path = self.setup_conf['dataset_files.data_path']
            if os.path.splitext(file_path)[1] == '.parquet':
                file_path = os.path.join(file_path, '*.parquet')
            data_files = glob(file_path)

            splitter = ListSplitter(
                data_files,
                valid_size=self.setup_conf['valid_size'],
                seed=self.setup_conf['valid_split_seed'],
            )
            logger.info(f'Prepared splits: '
                        f'{len(splitter.train)} files in train, {len(splitter.valid)} files in valid')

            self.train_dataset = ParquetDataset(
                splitter.train,
                post_processing=self.train_post_processing,
                shuffle_files=True,
            )
            self.valid_dataset = ParquetDataset(
                splitter.valid,
                post_processing=self.valid_post_processing,
                shuffle_files=False,
            )
        else:
            raise AttributeError(f'Unknown split strategy: {self.setup_conf.split_by}')

    @property
    def train_post_processing(self):
        return IterableChain(
            SeqLenFilter(min_seq_len=self.train_conf['min_seq_len']),
            TargetExtractor(target_col=self.col_id),
            FeatureFilter(drop_non_iterable=True),
            IterableShuffle(self.train_conf['buffer_size']),
            IterableSplittingDataset(split_strategy.create(**self.train_conf['split_strategy'])),
            IterableAugmentations(
                DropoutTrx(self.train_conf['trx_dropout']),
            ),
        )

    @property
    def valid_post_processing(self):
        return IterableChain(
            TargetExtractor(target_col=self.col_id),
            FeatureFilter(drop_non_iterable=True),
            IterableSplittingDataset(split_strategy.create(**self.valid_conf['split_strategy'])),
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=padded_collate,
            num_workers=self.train_conf['num_workers'],
            batch_size=self.train_conf['batch_size'],
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            collate_fn=padded_collate,
            num_workers=self.valid_conf['num_workers'],
            batch_size=self.valid_conf['batch_size'],
        )
