from typing import List, Dict
from functools import partial

import pytorch_lightning as pl
import torch


class PtlsDataModule(pl.LightningDataModule):
    """
    Generate an data loader. The data loader will return a batch of sequences.
    Args:
        data: List[Dict]
            The dataset
        batch_size: int. Default: 64.
            The number of samples (before splitting to subsequences) in each batch
        num_workers: int. Default: 0.
            The number of workers for the dataloader. 0 = single-process loader
        drop_last: bool. Default: False.
            Drop the last incomplete batch, if the dataset size is not divisible by the batch size

    Returns:
        DataLoader
    """
    def __init__(self,
                 train_data: List[Dict] = None,
                 train_batch_size: int = 64,
                 train_num_workers: int = 0,
                 train_drop_last: bool = False,
                 valid_data: List[Dict] = None,
                 valid_batch_size: int = 64,
                 valid_num_workers: int = 0,
                 valid_drop_last: bool = False,
                 test_data: List[Dict] = None, 
                 test_batch_size: int = 64,
                 test_num_workers: int = 0,
                 test_drop_last: bool = False,
                 ):

        super().__init__()
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_drop_last = train_drop_last
        self.valid_batch_size = valid_batch_size
        self.valid_num_workers = valid_num_workers
        self.valid_drop_last = valid_drop_last
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.test_drop_last = test_drop_last
        
        self.save_hyperparameters(ignore=['train_data', 'valid_data', 'test_data'])
        if self.hparams.valid_num_workers is None:
            self.hparams.valid_num_workers = self.hparams.train_num_workers
        if self.hparams.test_num_workers is None:
            self.hparams.test_num_workers = self.hparams.valid_num_workers

        if self.hparams.valid_batch_size is None:
            self.hparams.valid_batch_size = self.hparams.train_batch_size
        if self.hparams.test_batch_size is None:
            self.hparams.test_batch_size = self.hparams.valid_batch_size

        if train_data is not None:
            self.train_dataloader = partial(self.train_dl, train_data)
        if valid_data is not None:
            self.val_dataloader = partial(self.val_dl, valid_data)
        if test_data is not None:
            self.test_dataloader = partial(self.test_dl, test_data)

    def train_dl(self, train_data):
        return torch.utils.data.DataLoader(
            dataset=train_data,
            collate_fn=train_data.collate_fn,
            shuffle=not isinstance(train_data, torch.utils.data.IterableDataset),
            num_workers=self.hparams.train_num_workers,
            batch_size=self.hparams.train_batch_size,
            drop_last=self.hparams.train_drop_last,
        )

    def val_dl(self, valid_data):
        return torch.utils.data.DataLoader(
            dataset=valid_data,
            collate_fn=valid_data.collate_fn,
            shuffle=False,
            num_workers=self.hparams.valid_num_workers,
            batch_size=self.hparams.valid_batch_size,
            drop_last=self.hparams.valid_drop_last,
        )

    def test_dl(self, test_data):
        return torch.utils.data.DataLoader(
            dataset=test_data,
            collate_fn=test_data.collate_fn,
            shuffle=False,
            num_workers=self.hparams.test_num_workers,
            batch_size=self.hparams.test_batch_size,
            drop_last=self.hparams.test_drop_last,
        )
