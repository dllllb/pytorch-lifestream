import numpy as np
import torch
import pytorch_lightning as pl
from ptls.data_load import create_train_loader, create_validation_loader, TrxDataset
from ptls_tests.utils.data_generation import gen_trx_data


class RandomEventData(pl.LightningDataModule):
    def __init__(self, params, target_type='bin_cls'):
        super().__init__()
        self.hparams.update(params)
        self.target_type = target_type

    def train_dataloader(self):
        test_data = TrxDataset(
            gen_trx_data((torch.rand(1000) * 60 + 1).long(), target_type=self.target_type),
            y_dtype=np.int64 if self.target_type != 'regression' else np.float32,
        )
        train_loader = create_train_loader(test_data, self.hparams.train)
        return train_loader

    def val_dataloader(self):
        test_data = TrxDataset(
            gen_trx_data((torch.rand(200) * 60 + 1).long(), target_type=self.target_type),
            y_dtype=np.int64 if self.target_type != 'regression' else np.float32,
        )
        train_loader = create_validation_loader(test_data, self.hparams.valid)
        return train_loader
