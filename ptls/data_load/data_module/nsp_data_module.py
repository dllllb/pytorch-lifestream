"""
Creates data loaders for Next Sequence Prediction learning
Each record pass throw the sampler and splits into views of the main record
Each sample slits into left and right part.
Some right part are shuffled.
X - is pair of feature_arrays (left and right), sub-samples of main event sequences
Y - is left and right from she same sample.
    0 means left and right from different sample, 1 means left and right from the same sample

    Input:        -> Output
dict of arrays    ->  X,                                            Y
client_1          -> (client_1_smpl_1_left, client_1_smpl_1_right), 1
                     (client_1_smpl_2_left, client_1_smpl_2_right), 1
                     (client_1_smpl_3_left, client_3_smpl_1_right), 0
client_2          -> (client_2_smpl_1_left, client_2_smpl_3_right), 0
                     ...
"""
import logging
import random

import numpy as np
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ptls.data_load.augmentations.build_augmentations import build_augmentations
from ptls.data_load.augmentations.sequence_pair_augmentation import sequence_pair_augmentation
from ptls.data_load import IterableChain, augmentation_chain, padded_collate_wo_target
from ptls.data_load.iterable_processing.category_size_clip import CategorySizeClip
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.data_load.iterable_processing.iterable_shuffle import IterableShuffle
from ptls.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from ptls.data_load.list_splitter import ListSplitter
from ptls.data_load.parquet_dataset import ParquetDataset, ParquetFiles
from ptls.data_load.partitioned_dataset import PartitionedDataset, PartitionedDataFiles
from ptls.metric_learn.dataset import split_strategy, nested_list_to_flat_with_collate
from ptls.metric_learn.dataset.splitting_dataset import IterableSplittingDataset, MapSplittingDataset

logger = logging.getLogger(__name__)


def collate_nsp_pairs(batch):
    lefts = [left for left, _ in batch] * 2

    rights = [right for _, right in batch]
    rights_ = rights[:]
    random.shuffle(rights_)
    rights += rights_

    targets = torch.cat([
        torch.ones(len(batch), dtype=torch.int64),
        torch.zeros(len(batch), dtype=torch.int64),
    ])

    return (
        (
            padded_collate_wo_target(lefts),
            padded_collate_wo_target(rights)
        ),
        targets.float(),
    )


class NspDataModuleTrain(pl.LightningDataModule):
    def __init__(self, conf, pl_module):
        super().__init__()

        self._type = conf['type']
        assert self._type in ('map', 'iterable')

        self.setup_conf = conf['setup']
        self.train_conf = conf['train']
        self.valid_conf = conf['valid']

        self.col_id = self.setup_conf['col_id']
        self.category_names = pl_module.seq_encoder.category_names
        self.category_names.add('event_time')
        self.category_max_size = pl_module.seq_encoder.category_max_size

        self.train_dataset = None
        self.valid_dataset = None
        self._train_ids = None
        self._valid_ids = None

    def prepare_data(self):
        if 'dataset_files' in self.setup_conf:
            self.setup_iterable_files()
        elif 'dataset_parts' in self.setup_conf:
            self.setup_iterable_parts()
        else:
            raise AttributeError(f'Missing dataset definition. One of `dataset_parts` or `dataset_files` expected')

        if self._type == 'map':
            self.setup_map()

    def setup_iterable_files(self):
        if self.setup_conf['split_by'] == 'files':
            data_files = ParquetFiles(self.setup_conf['dataset_files.data_path']).data_files

            splitter = ListSplitter(
                data_files,
                valid_size=self.setup_conf['valid_size'],
                seed=self.setup_conf['valid_split_seed'],
            )
            logger.info(f'Prepared splits: '
                        f'{len(splitter.train)} files in train, {len(splitter.valid)} files in valid')

            self.train_dataset = ParquetDataset(
                splitter.train,
                post_processing=IterableChain(*self.build_iterable_processing('train')),
                shuffle_files=True if self._type == 'iterable' else False,
            )
            self.valid_dataset = ParquetDataset(
                splitter.valid,
                post_processing=IterableChain(*self.build_iterable_processing('valid')),
                shuffle_files=False,
            )

        else:
            raise AttributeError(f'Unknown split strategy: {self.setup_conf.split_by}')

    def setup_iterable_parts(self):
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
                post_processing=IterableChain(*self.build_iterable_processing('train')),
                shuffle_files=True if self._type == 'iterable' else False,
            )
            self.valid_dataset = PartitionedDataset(
                data_files.data_path, data_files.dt_parts, splitter.valid, col_id=self.col_id,
                post_processing=IterableChain(*self.build_iterable_processing('valid')),
                shuffle_files=False,
            )
        elif self.setup_conf['split_by'] == 'rows':
            raise NotImplementedError("Split by rows aren't supported for partitioned dataset")
        else:
            raise AttributeError(f'Unknown split strategy: {self.setup_conf.split_by}')

    def build_iterable_processing(self, part):
        if part == 'train':
            yield SeqLenFilter(min_seq_len=self.train_conf['min_seq_len'])

        yield FeatureFilter(keep_feature_names=self.category_names)
        yield CategorySizeClip(self.category_max_size)

        if self._type == 'iterable':
            # all processing in single chain
            if part == 'train':
                yield IterableShuffle(self.train_conf['buffer_size'])

            if part == 'train':
                split_strategy_conf = self.train_conf['split_strategy']
            else:
                split_strategy_conf = self.valid_conf['split_strategy']
            yield IterableSplittingDataset(split_strategy.create(**split_strategy_conf),
                                           self.build_augmentations(part))

    def build_augmentations(self, part):
        if part == 'train':
            a_chain = build_augmentations(self.train_conf['augmentations'])
        else:
            a_chain = build_augmentations(self.valid_conf['augmentations'])
        return augmentation_chain(a_chain, sequence_pair_augmentation)

    def setup_map(self):
        self.train_dataset = list(tqdm(iter(self.train_dataset)))
        logger.info(f'Loaded {len(self.train_dataset)} for train')
        self.valid_dataset = list(tqdm(iter(self.valid_dataset)))
        logger.info(f'Loaded {len(self.valid_dataset)} for valid')

        self.train_dataset = MapSplittingDataset(
            base_dataset=self.train_dataset,
            splitter=split_strategy.create(**self.train_conf['split_strategy']),
            a_chain=self.build_augmentations('train'),
        )
        self.valid_dataset = MapSplittingDataset(
            base_dataset=self.valid_dataset,
            splitter=split_strategy.create(**self.valid_conf['split_strategy']),
            a_chain=self.build_augmentations('valid'),
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=nested_list_to_flat_with_collate(collate_nsp_pairs),
            shuffle=False if self._type == 'iterable' else True,
            num_workers=self.train_conf['num_workers'],
            batch_size=self.train_conf['batch_size'],
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            collate_fn=nested_list_to_flat_with_collate(collate_nsp_pairs),
            num_workers=self.valid_conf['num_workers'],
            batch_size=self.valid_conf['batch_size'],
        )
