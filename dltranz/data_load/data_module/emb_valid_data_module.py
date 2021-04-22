import json
import logging

import pytorch_lightning as pl
import numpy as np
from embeddings_validation.file_reader import TargetFile, ID_TYPE_MAPPING
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dltranz.data_load import padded_collate_emb_valid, IterableChain, IterableAugmentations
from dltranz.data_load.augmentations.build_augmentations import build_augmentations
from dltranz.data_load.data_module.map_augmentation_dataset import MapAugmentationDataset
from dltranz.data_load.iterable_processing.category_size_clip import CategorySizeClip
from dltranz.data_load.iterable_processing.feature_filter import FeatureFilter
from dltranz.data_load.iterable_processing.feature_type_cast import FeatureTypeCast
from dltranz.data_load.iterable_processing.id_filter import IdFilter
from dltranz.data_load.iterable_processing.iterable_shuffle import IterableShuffle
from dltranz.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from dltranz.data_load.iterable_processing.target_join import TargetJoin
from dltranz.data_load.parquet_dataset import ParquetFiles, ParquetDataset


logger = logging.getLogger(__name__)


class EmbValidDataModule(pl.LightningDataModule):
    def __init__(self, conf, pl_module):
        super().__init__()

        self._type = conf['type']
        assert self._type in ('map', 'iterable')

        self.setup_conf = conf['setup']
        self.train_conf = conf['train']
        self.valid_conf = conf['valid']
        self.test_conf = conf.get('test', self.valid_conf)

        self.col_id = self.setup_conf['col_id']
        self.col_id_dtype = {
            'str': str,
            'int': int,
        }[self.setup_conf['col_id_dtype']]
        self.col_target = self.setup_conf['col_target']
        self.category_names = pl_module.seq_encoder.category_names
        self.category_names.add('event_time')
        self.category_max_size = pl_module.seq_encoder.category_max_size

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self._train_targets = None
        self._valid_targets = None
        self._test_targets = None

        self._fold_info = None

    def prepare_data(self):
        if 'dataset_files' in self.setup_conf:
            self.setup_iterable_files()
        elif 'dataset_parts' in self.setup_conf:
            self.setup_iterable_parts()
        else:
            raise AttributeError(f'Missing dataset definition. One of `dataset_parts` or `dataset_files` expected')

        if self._type == 'map':
            self.setup_map()

    def setup_iterable_files(self):
       train_data_files = ParquetFiles(self.setup_conf['dataset_files.train_data_path']).data_files
       test_data_files = ParquetFiles(self.setup_conf['dataset_files.test_data_path']).data_files

       self.train_dataset = ParquetDataset(
           train_data_files,
           shuffle_files=True if self._type == 'iterable' else False,
       )
       self.valid_dataset = ParquetDataset(
           train_data_files,
           shuffle_files=False,
       )
       self.test_dataset = ParquetDataset(
           test_data_files,
           shuffle_files=False,
       )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=padded_collate_emb_valid,
            shuffle=False if self._type == 'iterable' else True,
            num_workers=self.train_conf['num_workers'],
            batch_size=self.train_conf['batch_size'],
            drop_last=self.train_conf.get('drop_last', False)
        )

    def setup_map(self):
        self.train_dataset = list(tqdm(iter(self.train_dataset)))
        logger.info(f'Loaded {len(self.train_dataset)} for train')
        self.valid_dataset = list(tqdm(iter(self.valid_dataset)))
        logger.info(f'Loaded {len(self.valid_dataset)} for valid')
        self.test_dataset = list(tqdm(iter(self.test_dataset)))
        logger.info(f'Loaded {len(self.test_dataset)} for test')

        self.train_dataset = MapAugmentationDataset(
            base_dataset=self.train_dataset,
        )
        self.valid_dataset = MapAugmentationDataset(
            base_dataset=self.valid_dataset,
        )
        self.test_dataset = MapAugmentationDataset(
            base_dataset=self.test_dataset,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            collate_fn=padded_collate_emb_valid,
            num_workers=self.valid_conf['num_workers'],
            batch_size=self.valid_conf['batch_size'],
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=padded_collate_emb_valid,
            num_workers=self.test_conf['num_workers'],
            batch_size=self.test_conf['batch_size'],
        )
