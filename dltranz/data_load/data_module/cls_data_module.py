import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd

from dltranz.data_load import padded_collate, IterableChain, IterableAugmentations
from dltranz.data_load.augmentations.dropout_trx import DropoutTrx
from dltranz.data_load.iterable_processing.feature_filter import FeatureFilter
from dltranz.data_load.iterable_processing.id_filter import IdFilter
from dltranz.data_load.iterable_processing.iterable_shuffle import IterableShuffle
from dltranz.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from dltranz.data_load.iterable_processing.target_join import TargetJoin
from dltranz.data_load.parquet_dataset import ParquetFiles, ParquetDataset


class ClsDataModuleTrain(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()

        self._type = conf['type']
        assert self._type in ('map', 'iterable')

        self.setup_conf = conf['setup']
        self.train_conf = conf['train']
        self.valid_conf = conf['valid']
        self.test_conf = conf.get('test', self.valid_conf)

        self.col_id = self.setup_conf['col_id']
        self.col_target = self.setup_conf['col_target']

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self._train_targets = None
        self._valid_targets = None
        self._test_targets = None

    def setup(self, stage=None):
        if 'dataset_files' in self.setup_conf:
            self.setup_iterable_files()
        elif 'dataset_parts' in self.setup_conf:
            self.setup_iterable_parts()
        else:
            raise AttributeError(f'Missing dataset definition. One of `dataset_parts` or `dataset_files` expected')

        if self._type == 'map':
            self.setup_map()

    def setup_iterable_files(self):
        if self.setup_conf['split_by'] == 'external':
            data_files = ParquetFiles(self.setup_conf['dataset_files.data_path']).data_files

            self.read_external_splits()

            self.train_dataset = ParquetDataset(
                data_files,
                post_processing=IterableChain(*self.build_iterable_processing('train')),
                shuffle_files=True if self._type == 'iterable' else False,
            )
            self.valid_dataset = ParquetDataset(
                data_files,
                post_processing=IterableChain(*self.build_iterable_processing('valid')),
                shuffle_files=False,
            )
            self.test_dataset = ParquetDataset(
                data_files,
                post_processing=IterableChain(*self.build_iterable_processing('test')),
                shuffle_files=False,
            )

        else:
            raise AttributeError(f'Unknown split strategy: {self.setup_conf.split_by}')

    def read_external_splits(self):
        self._train_targets = pd.read_pickle(self.setup_conf['train_targets'])
        self._valid_targets = pd.read_pickle(self.setup_conf['valid_targets'])
        self._test_targets = pd.read_pickle(self.setup_conf['test_targets'])

    def build_iterable_processing(self, part):
        if 'dataset_files' in self.setup_conf and self.setup_conf['split_by'] == 'external':
            if part == 'train':
                yield IdFilter(id_col=self.col_id, relevant_ids=self._train_targets[self.col_id].values.tolist())
            elif part == 'valid':
                yield IdFilter(id_col=self.col_id, relevant_ids=self._valid_targets[self.col_id].values.tolist())
            elif part == 'test':
                yield IdFilter(id_col=self.col_id, relevant_ids=self._test_targets[self.col_id].values.tolist())
            else:
                raise AttributeError(f'Unknown part: {part}')

        if part == 'train':
            yield SeqLenFilter(min_seq_len=self.train_conf['min_seq_len'])

        if part == 'train':
            yield TargetJoin(self.col_id, self._train_targets[self.col_target].to_dict())
        elif part == 'valid':
            yield TargetJoin(self.col_id, self._valid_targets[self.col_target].to_dict())
        elif part == 'test':
            yield TargetJoin(self.col_id, self._test_targets[self.col_target].to_dict())
        else:
            raise AttributeError(f'Unknown part: {part}')

        yield FeatureFilter(drop_non_iterable=True)

        if self._type == 'iterable':
            # all processing in single chain
            if part == 'train':
                yield IterableShuffle(self.train_conf['buffer_size'])

            yield IterableAugmentations(*self.build_augmentations(part))

    def build_augmentations(self, part):
        if part == 'train':
            yield DropoutTrx(self.train_conf['trx_dropout'])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=padded_collate,
            shuffle=False if self._type == 'iterable' else True,
            num_workers=self.train_conf['num_workers'],
            batch_size=self.train_conf['batch_size'],
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            collate_fn=padded_collate,
            num_workers=self.valid_conf['num_workers'],
            batch_size=self.valid_conf['batch_size'],
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=padded_collate,
            num_workers=self.test_conf['num_workers'],
            batch_size=self.test_conf['batch_size'],
        )

