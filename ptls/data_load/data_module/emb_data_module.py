import logging

import pytorch_lightning as pl
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import List, Dict, Union
from tqdm.auto import tqdm

from ptls.data_load import IterableChain, padded_collate_wo_target
from ptls.data_load.augmentations.build_augmentations import build_augmentations
from ptls.data_load.data_module.coles_data_module import coles_collate_fn
from ptls.data_load.filter_dataset import FilterDataset
from ptls.data_load.parquet_dataset import ParquetFiles, ParquetDataset
from ptls.data_load.iterable_processing.category_size_clip import CategorySizeClip
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.data_load.iterable_processing.filter_non_array import FilterNonArray
from ptls.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch
from ptls.metric_learn.dataset import split_strategy
from ptls.metric_learn.dataset.splitting_dataset import MapSplittingDataset

logger = logging.getLogger(__name__)


r"""Generate a train data loader

Parameters
----------
data: List[Dict]
The dataset
min_seq_len: int. Default: 0.
The minimal length of sequences used for training. The shorter sequences would be skipped.
seq_split_strategy: str. Options: 'SampleSlices',  Default: 'SampleSlices'.
The strategy for splitting sequence to subsequences
split_count: int. Default: 5.
An amount of subsequences from one sequence.
split_cnt_min: int. Default: 1.
The minimal amount of events in each subsequence.
split_cnt_max: int. Default: 4000.
    The maximal amount of events in each subsequence.
num_workers: int. Default: 0.
The number of workers for the dataloader. 0 = single-process loader
train_batch_size: int. Default: 512.
The number of samples (before splitting to subsequences) in each batch
"""
def train_data_loader(
    data: List[Dict],
    min_seq_len: int = 0,
    seq_split_strategy: str = 'SampleSlices',
    split_count: int = 5,
    split_cnt_min: int = 1,
    split_cnt_max: int = 4000,
    drop_cols: List[str] = list(),
    num_workers: int = 0,
    batch_size: int = 512):

    dataset = FilterDataset(
        data,
        post_processing=IterableChain(
            SeqLenFilter(min_seq_len=min_seq_len),
            FilterNonArray(),
            ToTorch(),
            FeatureFilter(drop_feature_names=drop_cols)
        )
    )

    dataset = list(iter(dataset))

    split_strategy_params = {
        'split_strategy': seq_split_strategy,
        'split_count': split_count,
        'cnt_min': split_cnt_min,
        'cnt_max': split_cnt_max
    }

    dataset = MapSplittingDataset(
        base_dataset=dataset,
        splitter=split_strategy.create(**split_strategy_params),
        a_chain=build_augmentations({}),
    )

    return DataLoader(
        dataset=dataset,
        collate_fn=coles_collate_fn,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size
    )


r"""Generate an inference data loader

Parameters
----------
data: List[Dict]
The dataset
num_workers: int. Default: 0.
The number of workers for the dataloader. 0 = single-process loader
train_batch_size: int. Default: 512.
The number of samples (before splitting to subsequences) in each batch
"""
def inference_data_loader(
    data: List[Dict],
    num_workers: int = 0,
    batch_size: int = 512):

    dataset = FilterDataset(
        data,
        post_processing=IterableChain(
            FilterNonArray(),
            ToTorch()
        )
    )

    return DataLoader(
        dataset=dataset,
        collate_fn=padded_collate_wo_target,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
    )


class EmbeddingTrainDataModule(pl.LightningDataModule):
    r"""pytorch-lightning data module for unsupervised CoLES training

    Parameters
    ----------
     dataset: List[Dict]
        dataset
     test_size: float
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
     pl_module: pl.LightningModule
        The model (pl.LightningModule object).
     parquet_train_path: str
        Path to parquet files with train data.
     parquet_valid_path: str
        Path to parquet files with validation data.
     iterable: bool. Default: False.
        Whether to use iterable or map dataset.
     min_seq_len: int. Default: 0.
        The minimal length of sequences used for training. The shorter sequences would be skipped.
     augmentations: List. Default: w/0 augmentations.
        The list of augmentations to be applied to the samples during training.
     seq_split_strategy: str. Options: 'SampleSlices',  Default: 'SampleSlices'.
        The strategy for splitting sequence to subsequences
     split_count: int. Default: 5.
        An amount of subsequences from one sequence.
     split_cnt_min: int. Default: 1.
        The minimal amount of events in each subsequence.
     split_cnt_max: int. Default: 4000.
         The maximal amount of events in each subsequence.
     train_num_workers: int. Default: 0.
        The number of workers for train dataloader. 0 = single-process loader
     train_batch_size: int. Default: 512.
        The number of samples (before splitting to subsequences) in each batch during training.
     valid_num_workers: int. Default: 0.
        The number of workers for validation dataloader. 0 = single-process loader
     valid_batch_size: int. Default: 512.
        The number of samples (before splitting to subsequences) in each batch during validation.
     col_time: str. Default: 'event_time'
        The name of date and time column.
     category_names: List[str]. Default: None.
        The names of features (keys in dataset)
     category_max_size: Dict[str, int]. Default: None.
        The max value of categorical features (embedding matrix size)
     random_state : int. Default : 42.
        Controls the shuffling applied to the data before applying the split.
     """

    def __init__(self,
                 dataset: List[Dict] = None,
                 pl_module: pl.LightningModule = None,
                 parquet_train_path: str = None,
                 parquet_valid_path: str = None,
                 test_size: float = 0.1,
                 iterable: bool = False,
                 min_seq_len: int = 0,
                 augmentations: List = None,
                 seq_split_strategy: str = 'SampleSlices',
                 split_count: int = 5,
                 split_cnt_min: int = 1,
                 split_cnt_max: int = 4000,
                 train_num_workers: int = 0,
                 train_batch_size: int = 512,
                 valid_num_workers: int = 0,
                 valid_batch_size: int = 512,
                 category_names: List[str] = None,
                 category_max_size: Dict[str, int] = None,
                 col_time: str = 'event_time',
                 drop_cols: List[str] = [],
                 random_state: int = 42):

        required_s = 'Necessary to supply `dataset` or `parquet_train_path` and `parquet_valid_path`'
        assert dataset or (parquet_train_path and parquet_valid_path), required_s
        if dataset and (parquet_train_path or parquet_valid_path):
            print('`dataset` is supplied, `parquet_train_path` and `parquet_valid_path` will be ignored')

        super().__init__()
        self.drop_cols = drop_cols

        self.split_strategy_params = {
            'split_strategy': seq_split_strategy,
            'split_count': split_count,
            'cnt_min': split_cnt_min,
            'cnt_max': split_cnt_max
        }

        self.train_num_workers = train_num_workers
        self.train_batch_size = train_batch_size
        self.valid_num_workers = valid_num_workers
        self.valid_batch_size = valid_batch_size
        self.min_seq_len = min_seq_len

        self._type = 'iterable' if iterable else 'map'
        self.col_time = col_time

        self.category_names = category_names
        if self.category_names is not None:
            self.category_names.add(col_time)
        self.category_max_size = category_max_size

        if dataset:
            self.train_data, self.valid_data = train_test_split(dataset,
                                                                test_size=test_size,
                                                                random_state=random_state)
            self.dataset_cls = FilterDataset
        else:
            self.train_data = ParquetFiles(parquet_train_path).data_files
            self.valid_data = ParquetFiles(parquet_valid_path).data_files
            self.dataset_cls = ParquetDataset

        self.train_dataset = None
        self.valid_dataset = None

    def prepare_data(self):
        self.train_dataset = self.dataset_cls(
            self.train_data,
            post_processing=IterableChain(*self.build_processing('train'))
        )

        self.valid_dataset = self.dataset_cls(
            self.valid_data,
            post_processing=IterableChain(*self.build_processing('valid'))
        )

        if self._type == 'map':
            self.setup_map()

        elif self._type == 'iterable':
            raise NotImplementedError("Not implemented yet")

        else:
            raise AttributeError(f'Should be unreachable')

    def build_augmentations(self, part):
        if part == 'train':
            return build_augmentations([])

        if part == 'valid':
            return build_augmentations([])

    def build_processing(self, part):
        if part == 'train':
            yield SeqLenFilter(min_seq_len=self.min_seq_len)
        yield ToTorch()
        yield FeatureFilter(keep_feature_names=self.category_names,
                            drop_feature_names=self.drop_cols)
        yield CategorySizeClip(self.category_max_size, 1)

    def setup_iterable(self):
        self.train_dataset = MapSplittingDataset(
            base_dataset=self.train_dataset,
            splitter=split_strategy.create(**self.split_strategy_params),
            a_chain=self.build_augmentations('train'),
        )
        self.valid_dataset = MapSplittingDataset(
            base_dataset=self.valid_dataset,
            splitter=split_strategy.create(**self.split_strategy_params),
            a_chain=self.build_augmentations('valid'),
        )

    def setup_map(self):
        self.train_dataset = list(tqdm(iter(self.train_dataset)))
        logger.info(f'Loaded {len(self.train_dataset)} for train')
        self.valid_dataset = list(tqdm(iter(self.valid_dataset)))
        logger.info(f'Loaded {len(self.valid_dataset)} for valid')

        self.train_dataset = MapSplittingDataset(
            base_dataset=self.train_dataset,
            splitter=split_strategy.create(**self.split_strategy_params),
            a_chain=self.build_augmentations('train'),
        )
        self.valid_dataset = MapSplittingDataset(
            base_dataset=self.valid_dataset,
            splitter=split_strategy.create(**self.split_strategy_params),
            a_chain=self.build_augmentations('valid'),
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=coles_collate_fn,
            shuffle=False if self._type == 'iterable' else True,
            num_workers=self.train_num_workers,
            batch_size=self.train_batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            collate_fn=coles_collate_fn,
            num_workers=self.valid_num_workers,
            batch_size=self.valid_batch_size,
        )
