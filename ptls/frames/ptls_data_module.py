from functools import partial

import pytorch_lightning as pl
import torch


class PtlsDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data=None,
                 valid_data=None,
                 test_data=None,
                 ):

        super().__init__()
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
