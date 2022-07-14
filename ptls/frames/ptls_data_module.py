import torch
import pytorch_lightning as pl


class PtlsDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data=None,
                 train_batch_size=1,
                 train_num_workers=0,
                 train_drop_last=False,
                 valid_data=None,
                 valid_batch_size=None,
                 valid_num_workers=None,
                 valid_drop_last=False,
                 test_data=None,
                 test_batch_size=None,
                 test_num_workers=None,
                 test_drop_last=False,
                 ):

        super().__init__()

        if valid_num_workers is None:
            valid_num_workers = train_num_workers
        if test_num_workers is None:
            test_num_workers = valid_num_workers

        if valid_batch_size is None:
            valid_batch_size = train_batch_size
        if test_batch_size is None:
            test_batch_size = valid_batch_size

        def train_dataloader():
            return torch.utils.data.DataLoader(
                dataset=train_data,
                collate_fn=train_data.collate_fn,
                shuffle=not isinstance(train_data, torch.utils.data.IterableDataset),
                num_workers=train_num_workers,
                batch_size=train_batch_size,
                drop_last=train_drop_last,
            )

        def val_dataloader():
            return torch.utils.data.DataLoader(
                dataset=valid_data,
                collate_fn=valid_data.collate_fn,
                shuffle=False,
                num_workers=valid_num_workers,
                batch_size=valid_batch_size,
                drop_last=valid_drop_last,
            )

        def test_dataloader():
            return torch.utils.data.DataLoader(
                dataset=test_data,
                collate_fn=test_data.collate_fn,
                shuffle=False,
                num_workers=test_num_workers,
                batch_size=test_batch_size,
                drop_last=test_drop_last,
            )

        if train_data is not None:
            self.train_dataloader = train_dataloader
        if valid_data is not None:
            self.val_dataloader = val_dataloader
        if test_data is not None:
            self.test_dataloader = test_dataloader
