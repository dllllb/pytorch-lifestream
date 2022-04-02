import pytorch_lightning as pl
from dltranz.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from dltranz.data_load.iterable_processing.feature_filter import FeatureFilter
from dltranz.data_load.iterable_processing.category_size_clip import CategorySizeClip
from dltranz.data_load.iterable_processing.target_move import TargetMove
from dltranz.data_load.iterable_processing.to_torch_tensor import toTorchTensor
from dltranz.data_load import IterableChain
from torch.utils.data import DataLoader
from dltranz.data_load import padded_collate
from typing import List, Dict
from sklearn.model_selection import train_test_split


class SeqToTargetDatamodule(pl.LightningDataModule):
    def __init__(self,
                 dataset: List[dict],
                 pl_module: pl.LightningModule,
                 min_seq_len: int = 0,
                 val_size: float = 0.05,
                 train_num_workers: int = 0,
                 train_batch_size: int = 256,
                 val_num_workers: int = 0,
                 val_batch_size: int = 256,
                 random_state: int = 42,
                 target_col: str = 'target'):

        super().__init__()
        self.dataset_train, self.dataset_val = train_test_split(dataset,
                                                                test_size=val_size,
                                                                random_state=random_state)
        self.min_seq_len=min_seq_len
        self.train_num_workers = train_num_workers
        self.train_batch_size = train_batch_size
        self.val_num_workers = val_num_workers
        self.val_batch_size = val_batch_size
        self.keep_features = pl_module.seq_encoder.category_names
        self.keep_features.add('event_time')
        self.category_max_size = pl_module.seq_encoder.category_max_size
        self.target_col = target_col
        self.post_proc = IterableChain(*self.build_iterable_processing())

    def prepare_data(self):
        self.dataset_train = list(self.post_proc(iter(self.dataset_train)))
        self.dataset_val = list(self.post_proc(iter(self.dataset_val)))

    def build_iterable_processing(self):
        yield SeqLenFilter(min_seq_len=self.min_seq_len)
        yield toTorchTensor()
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
            dataset=self.dataset_val,
            collate_fn=padded_collate,
            num_workers=self.val_num_workers,
            batch_size=self.val_batch_size
        )

