import pytorch_lightning as pl
from ptls.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.data_load.iterable_processing.category_size_clip import CategorySizeClip
from ptls.data_load.iterable_processing.target_move import TargetMove
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch
from ptls.data_load import IterableChain
from torch.utils.data import DataLoader
from ptls.data_load import padded_collate
from typing import List, Dict
from sklearn.model_selection import train_test_split


class SeqToTargetDatamodule(pl.LightningDataModule):
    r"""pytorch-lightning data module for supervised training
    Parameters
    ----------
     dataset: List[Dict]
        dataset
     pl_module: pl.LightningModule
        Pytorch-lightning module used for training seq_encoder.
     min_seq_len: int. Default: 0.
        The minimal length of sequences used for training. The shorter sequences would be skipped.
     valid_size: float. Default: 0.05.
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
     train_num_workers: int. Default: 0.
        The number of workers for train dataloader. 0 = single-process loader
     train_batch_size: int. Default: 256.
        The number of samples (before splitting to subsequences) in each batch during training.
     valid_num_workers: int. Default: 0.
        The number of workers for validation dataloader. 0 = single-process loader       
     valid_batch_size: int. Default: 256.
        The number of samples (before splitting to subsequences) in each batch during validation.
     target_col: str. Default: 'target'.
        The name of target column.
     random_state : int. Default : 42.
        Controls the shuffling applied to the data before applying the split.
     """
    def __init__(self,
                 dataset: List[dict],
                 pl_module: pl.LightningModule,
                 min_seq_len: int = 0,
                 valid_size: float = 0.05,
                 train_num_workers: int = 0,
                 train_batch_size: int = 256,
                 valid_num_workers: int = 0,
                 valid_batch_size: int = 256,
                 target_col: str = 'target',
                 random_state: int = 42):

        super().__init__()
        self.dataset_train, self.dataset_valid = train_test_split(dataset,
                                                                  test_size=valid_size,
                                                                  random_state=random_state)
        self.min_seq_len=min_seq_len
        self.train_num_workers = train_num_workers
        self.train_batch_size = train_batch_size
        self.valid_num_workers = valid_num_workers
        self.valid_batch_size = valid_batch_size
        self.keep_features = pl_module.seq_encoder.category_names
        self.keep_features.add('event_time')
        self.category_max_size = pl_module.seq_encoder.category_max_size
        self.target_col = target_col
        self.post_proc = IterableChain(*self.build_iterable_processing())

    def prepare_data(self):
        self.dataset_train = list(self.post_proc(iter(self.dataset_train)))
        self.dataset_valid = list(self.post_proc(iter(self.dataset_valid)))

    def build_iterable_processing(self):
        yield SeqLenFilter(min_seq_len=self.min_seq_len)
        yield ToTorch()
        yield TargetMove(self.target_col)
        yield FeatureFilter(keep_feature_names=self.keep_features)
        yield CategorySizeClip(self.category_max_size)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            collate_fn=padded_collate,
            num_workers=self.train_num_workers,
            batch_size=self.train_batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_valid,
            collate_fn=padded_collate,
            num_workers=self.valid_num_workers,
            batch_size=self.valid_batch_size
        )

