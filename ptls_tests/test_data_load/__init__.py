import numpy as np
import torch
import pytorch_lightning as pl
from ptls.data_load import create_train_loader, create_validation_loader, TrxDataset


def gen_trx_data(lengths, target_share=.5, target_type='bin_cls'):
    n = len(lengths)
    if target_type == 'bin_cls':
        targets = (torch.rand(n) >= target_share).long()
    elif target_type == 'multi_cls':
        targets = torch.randint(0, 4, (n,))
    elif target_type == 'regression':
        targets = torch.randn(n)
    else:
        raise AttributeError(f'Unknown target_type: {target_type}')

    samples = list()
    for target, length in zip(targets, lengths):
        s = dict()
        s['trans_type'] = (torch.rand(length)*10 + 1).long()
        s['mcc_code'] = (torch.rand(length) * 20 + 1).long()
        s['amount'] = (torch.rand(length) * 1000 + 1).long()

        samples.append({'feature_arrays': s, 'target': target})

    return samples


class RandomEventData(pl.LightningDataModule):
    def __init__(self, params, target_type='bin_cls'):
        super().__init__()
        self.hparams.update(params)
        self.target_type = target_type

    def train_dataloader(self):
        test_data = TrxDataset(
            gen_trx_data((torch.rand(1000)*60+1).long(), target_type=self.target_type),
            y_dtype=np.int64 if self.target_type != 'regression' else np.float32,
        )
        train_loader = create_train_loader(test_data, self.hparams.train)
        return train_loader

    def val_dataloader(self):
        test_data = TrxDataset(
            gen_trx_data((torch.rand(200)*60+1).long(), target_type=self.target_type),
            y_dtype=np.int64 if self.target_type != 'regression' else np.float32,
        )
        train_loader = create_validation_loader(test_data, self.hparams.valid)
        return train_loader
