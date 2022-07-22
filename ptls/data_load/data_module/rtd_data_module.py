"""
Creates data loaders for contrastive learning
Each record is features from main record
X - is feature_arrays, samples of event sequences
Y - is Id of main record, not used

    Input:        -> Output
dict of arrays    ->  X,                 Y
client_1          -> client_1_features, client_1_id
client_2          -> client_2_features, client_2_id
"""
import logging
from functools import partial

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ptls.data_load import IterableAugmentations, IterableChain, padded_collate_wo_target
from ptls.data_load.augmentations.build_augmentations import build_augmentations
from ptls.data_load.data_module.map_augmentation_dataset import MapAugmentationDataset
from ptls.data_load.iterable_processing.category_size_clip import CategorySizeClip
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.data_load.iterable_processing.iterable_shuffle import IterableShuffle
from ptls.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from ptls.data_load.list_splitter import ListSplitter
from ptls.data_load.datasets.parquet_dataset import ParquetDataset, ParquetFiles
from ptls.data_load.partitioned_dataset import PartitionedDataset, PartitionedDataFiles
from ptls.data_load.padded_batch import PaddedBatch

logger = logging.getLogger(__name__)


def collate_rtd_batch(batch, replace_prob, skip_first=0):
    padded_batch = padded_collate_wo_target(batch)

    new_x, lengths, mask = padded_batch.payload, padded_batch.seq_lens, padded_batch.seq_len_mask

    to_replace = torch.bernoulli(mask * replace_prob).bool()
    to_replace[:, :skip_first] = False

    sampled_trx_ids = torch.multinomial(
        mask.flatten().float(),
        num_samples=to_replace.sum().item(),
        replacement=True
    )

    to_replace_flatten = to_replace.flatten()
    for k, v in new_x.items():
        v.flatten()[to_replace_flatten] = v.flatten()[sampled_trx_ids]

    return PaddedBatch(new_x, lengths), to_replace.long().flatten()[mask.flatten().bool()]


class RtdDataModuleTrain(pl.LightningDataModule):
    def __init__(self, type, setup, train, valid, pl_module, replace_token=False):
        super().__init__()

        self._type = type
        assert self._type in ('map', 'iterable')

        self.setup_conf = setup
        self.train_conf = train
        self.valid_conf = valid
        self.replace_token_conf = replace_token

        self.col_id = self.setup_conf.col_id
        self.category_names = pl_module.seq_encoder.category_names
        self.category_names.add('event_time')
        self.category_max_size = pl_module.seq_encoder.category_max_size

        self.train_dataset = None
        self.valid_dataset = None
        self._train_ids = None
        self._valid_ids = None

    def prepare_data(self, stage=None):
        if 'dataset_files' in self.setup_conf:
            self.setup_iterable_files()
        elif 'dataset_parts' in self.setup_conf:
            self.setup_iterable_parts()
        else:
            raise AttributeError(f'Missing dataset definition. One of `dataset_parts` or `dataset_files` expected')

        if self._type == 'map':
            self.setup_map()

    def setup_iterable_files(self):
        if self.setup_conf.split_by == 'files':
            data_files = ParquetFiles(self.setup_conf.dataset_files.data_path).data_files

            splitter = ListSplitter(
                data_files,
                valid_size=self.setup_conf.valid_size,
                seed=self.setup_conf.valid_split_seed,
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
        data_files = PartitionedDataFiles(**self.setup_conf.dataset_parts)

        if self.setup_conf.split_by == 'hash_id':
            splitter = ListSplitter(
                data_files.hash_parts,
                valid_size=self.setup_conf.valid_size,
                seed=self.setup_conf.valid_split_seed,
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
        elif self.setup_conf.split_by == 'rows':
            raise NotImplementedError("Split by rows aren't supported for partitioned dataset")
        else:
            raise AttributeError(f'Unknown split strategy: {self.setup_conf.split_by}')

    def build_iterable_processing(self, part):
        if part == 'train':
            yield SeqLenFilter(min_seq_len=self.train_conf.min_seq_len)

        yield FeatureFilter(keep_feature_names=self.category_names)
        yield CategorySizeClip(self.category_max_size)

        if self._type == 'iterable':
            # all processing in single chain
            if part == 'train':
                yield IterableShuffle(self.train_conf.buffer_size)

            yield IterableAugmentations(self.build_augmentations(part))

    def build_augmentations(self, part):
        if part == 'train':
            return build_augmentations(self.train_conf.augmentations)

        if part == 'valid':
            return build_augmentations(self.valid_conf.augmentations)

    def setup_map(self):
        self.train_dataset = list(tqdm(iter(self.train_dataset)))
        logger.info(f'Loaded {len(self.train_dataset)} for train')
        self.valid_dataset = list(tqdm(iter(self.valid_dataset)))
        logger.info(f'Loaded {len(self.valid_dataset)} for valid')

        self.train_dataset = MapAugmentationDataset(
            base_dataset=self.train_dataset,
            a_chain=self.build_augmentations('train'),
        )
        self.valid_dataset = MapAugmentationDataset(
            base_dataset=self.valid_dataset,
            a_chain=self.build_augmentations('valid'),
        )

    def train_dataloader(self):
        collate_fn = partial(
            collate_rtd_batch,
            replace_prob=self.replace_token_conf.replace_prob,
            skip_first=self.replace_token_conf.skip_first,
        )

        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=collate_fn,
            shuffle=False if self._type == 'iterable' else True,
            num_workers=self.train_conf.num_workers,
            batch_size=self.train_conf.batch_size,
        )

    def val_dataloader(self):
        collate_fn = partial(
            collate_rtd_batch,
            replace_prob=self.replace_token_conf.replace_prob,
            skip_first=self.replace_token_conf.skip_first,
        )

        return DataLoader(
            dataset=self.valid_dataset,
            collate_fn=collate_fn,
            num_workers=self.valid_conf.num_workers,
            batch_size=self.valid_conf.batch_size,
        )
