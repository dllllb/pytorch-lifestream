import warnings
from typing import List

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ptls.data_load import IterableChain
from ptls.data_load import padded_collate
from ptls.data_load.iterable_processing import FeatureFilter
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.data_load.iterable_processing.target_move import TargetMove
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch



class SeqToTargetDatamodule(pl.LightningDataModule):
   """
   PyTorch Lightning data module for supervised training.

   Args:
      dataset: The dataset.
      min_seq_len (int, optional): The minimal length of sequences used for training. Sequences shorter than this will be skipped. Defaults to 0.
      valid_size (float, optional): The proportion of the dataset to include in the validation split. Should be between 0.0 and 1.0. Defaults to 0.05.
      train_num_workers (int, optional): The number of workers for the training DataLoader. 0 means single-process loader. Defaults to 0.
      train_batch_size (int, optional): The number of samples in each batch during training. Defaults to 256.
      valid_num_workers (int, optional): The number of workers for the validation DataLoader. 0 means single-process loader. Defaults to 0.
      valid_batch_size (int, optional): The number of samples in each batch during validation. Defaults to 256.
      target_col (str, optional): The name of the target column. Defaults to 'target'.
      random_state (int, optional): Controls the shuffling applied to the data before splitting. Defaults to 42.

   """

   def __init__(self,
               dataset: List[dict],
               min_seq_len: int = 0,
               valid_size: float = 0.05,
               train_num_workers: int = 0,
               train_batch_size: int = 256,
               valid_num_workers: int = 0,
               valid_batch_size: int = 256,
               target_col: str = 'target',
               random_state: int = 42):
      warnings.warn('Use `ptls.frames.PtlsDataModule` with `ptls.frames.supervised.SeqToTargetDataset` '
                     'or `ptls.frames.supervised.SeqToTargetIterableDataset`',
                     DeprecationWarning)

      super().__init__()
      self.dataset_train, self.dataset_valid = train_test_split(dataset,
                                                               test_size=valid_size,
                                                               random_state=random_state)
      self.min_seq_len = min_seq_len
      self.train_num_workers = train_num_workers
      self.train_batch_size = train_batch_size
      self.valid_num_workers = valid_num_workers
      self.valid_batch_size = valid_batch_size
      self.target_col = target_col
      self.post_proc = IterableChain(*self.build_iterable_processing())

   def prepare_data(self):
      self.dataset_train = list(self.post_proc(iter(self.dataset_train)))
      self.dataset_valid = list(self.post_proc(iter(self.dataset_valid)))

   def build_iterable_processing(self):
      yield SeqLenFilter(min_seq_len=self.min_seq_len)
      yield ToTorch()
      yield TargetMove(self.target_col)
      yield FeatureFilter(drop_non_iterable=True)

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

   def get_test_dataloader(self, data, num_workers=0, batch_size=128):
      return DataLoader(
         dataset=list(self.post_proc(iter(data))),
         collate_fn=padded_collate,
         num_workers=num_workers,
         batch_size=batch_size,
      )
