"""
Creates data loaders for contrastive learning
Each record pass throw the sampler and splits into views of the main record
X - is feature_arrays, sub-samples of main event sequences
Y - is Id of main record, for positive item identification

    Input:        -> Output
dict of arrays    ->  X,                 Y
client_1          -> client_1_smpl_1, client_1_id
                     client_1_smpl_2, client_1_id
                     client_1_smpl_3, client_1_id
client_2          -> client_2_smpl_1, client_2_id
                     client_2_smpl_2, client_2_id
                     client_2_smpl_3, client_2_id
"""
import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dltranz.data_load import padded_collate, IterableAugmentations, IterableChain
from dltranz.data_load.augmentations.dropout_trx import DropoutTrx
from dltranz.data_load.iterable_processing.iterable_shuffle import IterableShuffle
from dltranz.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from dltranz.data_load.iterable_processing.target_extractor import TargetExtractor
from dltranz.data_load.list_splitter import ListSplitter
from dltranz.data_load.partitioned_dataset import PartitionedDataset, PartitionedDataFiles
from dltranz.metric_learn.dataset import split_strategy
from dltranz.metric_learn.dataset.splitting_dataset import IterableSplittingDataset

logger = logging.getLogger(__name__)


class ColesDataModulePartitionedIterableTrain(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.setup_conf = conf['setup']
        self.train_conf = conf['train']
        self.valid_conf = conf['valid']

        self.col_id = self.setup_conf['col_id']

        self.data_path = None
        self.dt_train = None
        self.dt_valid = None
        self.hash_train = None
        self.hash_valid = None

    def setup(self, stage=None):
        data_files = PartitionedDataFiles(**self.setup_conf['dataset_parts'])

        self.data_path = data_files.data_path
        if self.setup_conf['split_by'] == 'hash_id':
            splitter = ListSplitter(
                data_files.hash_parts,
                valid_size=self.setup_conf['valid_size'],
                seed=self.setup_conf['valid_split_seed'],
            )
            self.dt_train = data_files.dt_parts
            self.dt_valid = data_files.dt_parts
            self.hash_train = splitter.train
            self.hash_valid = splitter.valid

            logger.info(f'Prepared splits: '
                        f'{len(self.dt_train)} parts in dt_train, {len(self.dt_valid)} parts in dt_valid, '
                        f'{len(self.hash_train)} parts in hash_train, {len(self.hash_valid)} parts in hash_valid')
        else:
            raise AttributeError(f'Unknown split strategy: {self.setup_conf.split_by}')

    def train_dataloader(self):
        train_post_processing = IterableChain(
            SeqLenFilter(min_seq_len=self.train_conf['min_seq_len']),
            TargetExtractor(target_col=self.col_id),
            IterableShuffle(self.train_conf['buffer_size']),
            IterableSplittingDataset(split_strategy.create(**self.train_conf['split_strategy'])),
            IterableAugmentations(
                DropoutTrx(self.train_conf['trx_dropout']),
            ),
        )
        train_dataset = PartitionedDataset(
            self.data_path, self.dt_train, self.hash_train, col_id=self.col_id,
            post_processing=train_post_processing,
            shuffle_files=True,
        )
        logger.debug(f'train_dataset: {train_dataset}, {self.data_path}, {self.dt_train}, {self.hash_train}')
        return DataLoader(
            dataset=train_dataset,
            collate_fn=padded_collate,
            num_workers=self.train_conf['num_workers'],
            batch_size=self.train_conf['batch_size'],
        )

    def val_dataloader(self):
        valid_post_processing = IterableChain(
            TargetExtractor(target_col=self.col_id),
            IterableSplittingDataset(split_strategy.create(**self.valid_conf['split_strategy'])),
        )
        valid_dataset = PartitionedDataset(
            self.data_path, self.dt_valid, self.hash_valid, col_id=self.col_id,
            post_processing=valid_post_processing,
            shuffle_files=False,
        )
        return DataLoader(
            dataset=valid_dataset,
            collate_fn=padded_collate,
            num_workers=self.valid_conf['num_workers'],
            batch_size=self.valid_conf['batch_size'],
        )
