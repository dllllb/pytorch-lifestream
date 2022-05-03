import numpy as np
import torch
import pytorch_lightning as pl
from ptls.data_load import create_train_loader, create_validation_loader, TrxDataset


def gen_trx_data(lengths, target_share=.5):
    n = len(lengths)
    targets = (torch.rand(n) >= target_share).long()
    samples = list()
    for target, length in zip(targets, lengths):
        s = dict()
        s['trans_type'] = (torch.rand(length)*10 + 1).long()
        s['mcc_code'] = (torch.rand(length) * 20 + 1).long()
        s['amount'] = (torch.rand(length) * 1000 + 1).long()

        samples.append({'feature_arrays': s, 'target': target})

    return samples


class RandomEventData(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.hparams.update(params)

    def train_dataloader(self):
        test_data = TrxDataset(gen_trx_data((torch.rand(1000)*60+1).long()), y_dtype=np.int64)
        train_loader = create_train_loader(test_data, self.hparams['train'])
        return train_loader

    def val_dataloader(self):
        test_data = TrxDataset(gen_trx_data((torch.rand(200)*60+1).long()), y_dtype=np.int64)
        train_loader = create_validation_loader(test_data, self.hparams['valid'])
        return train_loader
