import ast, glob, json, random, warnings
import logging

import pytorch_lightning as pl
import numpy as np
import pandas as pd
from embeddings_validation.file_reader import TargetFile
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from ptls.data_load import padded_collate, padded_collate_distribution_target, IterableChain, IterableAugmentations
from ptls.data_load.augmentations.build_augmentations import build_augmentations
from ptls.data_load.data_module.map_augmentation_dataset import MapAugmentationDataset
from ptls.data_load.iterable_processing import FeatureFilter, FeatureTypeCast, IdFilter, IterableShuffle, SeqLenFilter, TargetJoin, TargetExtractor
from ptls.data_load.datasets.parquet_dataset import ParquetFiles, ParquetDataset
from ptls.data_load.utils import collate_target

logger = logging.getLogger(__name__)


class ClsDataModuleTrain(pl.LightningDataModule):
    def __init__(self,
                 type,
                 setup,
                 train,
                 valid,
                 pl_module,
                 fold_id,
                 test=None,
                 distribution_target_task=False,
                 distribution_target_size=None,
                 ):
        warnings.warn('Use `ptls.frames.PtlsDataModule` with `ptls.frames.supervised.SeqToTargetDataset` '
                      'or `ptls.frames.supervised.SeqToTargetIterableDataset`',
                      DeprecationWarning)
        super().__init__()

        self.fold_id = fold_id
        self._type = type
        assert self._type in ('map', 'iterable')

        self.distribution_target_task = distribution_target_task
        if distribution_target_size is None:
            self.y_function = int if not self.distribution_target_task else \
                lambda x: np.array(ast.literal_eval(x), dtype=object)
        elif distribution_target_size == 0:
            self.y_function = np.float32
        elif isinstance(distribution_target_size, int):
            self.y_function = lambda x: collate_target(ast.literal_eval(x), distribution_target_size)
        else:
            raise AttributeError(f"Undefined distribution target size: {distribution_target_size}")

        self.setup_conf = setup
        self.train_conf = train
        self.valid_conf = valid
        self.test_conf = test if test else self.valid_conf

        self.col_id = self.setup_conf.col_id
        self.col_id_dtype = {
            'str': str,
            'int': int,
        }[self.setup_conf.col_id_dtype]
        self.col_target = self.setup_conf.col_target
        self.category_names = pl_module.seq_encoder.category_names
        self.category_names.add('event_time')
        self.category_max_size = pl_module.seq_encoder.category_max_size

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self._train_targets = None
        self._valid_targets = None
        self._test_targets = None
        self._predict_targets = None

        self._fold_info = None
        self.data_is_prepared = False
        self.do_test = "test_data_path" in setup.dataset_files

    def prepare_data(self):
        if not self.data_is_prepared:
            if 'dataset_files' in self.setup_conf:
                self.setup_iterable_files()
            elif 'dataset_parts' in self.setup_conf:
                self.setup_iterable_parts()
            else:
                raise AttributeError(f'Missing dataset definition. One of `dataset_parts` or `dataset_files` expected')
            if self._type == 'map':
                self.setup_map()
            self.data_is_prepared = True

    def setup_iterable_files(self):
        if self.setup_conf.get('split_by', None) == 'embeddings_validation':

            if (self.setup_conf.get('use_files_partially', None)):
                n_train = len(glob.glob(self.setup_conf.dataset_files.train_data_path + "/*.parquet"))
                ixes = list(range(n_train))
                train_ixes, test_ixes = train_test_split(ixes, test_size=int(n_train * (1 - self.setup_conf.train_part)), shuffle=True)
                train_data_files = ParquetFiles(self.setup_conf.dataset_files.train_data_path, train_ixes).data_files
                if self.setup_conf.same_file_for_test:
                    test_ixes = random.sample(test_ixes, int(n_train * self.setup_conf.test_part))
                    test_data_files = ParquetFiles(self.setup_conf.dataset_files.train_data_path, test_ixes).data_files
                else:
                    n_test = len(glob.glob(self.setup_conf.dataset_files.test_data_path + "/*.parquet"))
                    test_ixes = random.sample(ixes, int(n_test * self.setup_conf.test_part))
                    test_data_files = ParquetFiles(self.setup_conf.dataset_files.test_data_path, test_ixes).data_files
            else:
                train_data_files = ParquetFiles(self.setup_conf.dataset_files.train_data_path).data_files
                test_data_files = ParquetFiles(self.setup_conf.dataset_files.test_data_path).data_files if self.do_test else []

            self.read_external_splits()
            self.train_dataset = ParquetDataset(
                train_data_files,
                post_processing=IterableChain(*self.build_iterable_processing('train')),
                shuffle_files=True if self._type == 'iterable' else False,
            )
            self.valid_dataset = ParquetDataset(
                train_data_files,
                post_processing=IterableChain(*self.build_iterable_processing('valid')),
                shuffle_files=False,
            )
            if self.do_test:
                self.test_dataset = ParquetDataset(
                    test_data_files,
                    post_processing=IterableChain(*self.build_iterable_processing('test')),
                    shuffle_files=False,
                )
            self.predict_dataset = ParquetDataset(
                train_data_files + test_data_files,
                post_processing=IterableChain(*self.build_iterable_processing('predict')),
                shuffle_files=False,
            )

        else:
            raise AttributeError(f'Unknown split strategy: {self.setup_conf.split_by}')

    def setup_iterable_parts(self):
        raise NotImplementedError()

    def read_external_splits(self):
        if self.setup_conf.split_by == 'embeddings_validation':
            with open(self.setup_conf.fold_info, 'r') as f:
                self._fold_info = json.load(f)

        current_fold = self._fold_info[self.fold_id]
        self._train_targets = TargetFile.load(current_fold['train']['path']).df
        self._valid_targets = TargetFile.load(current_fold['valid']['path']).df
        if self.do_test:
            self._test_targets = TargetFile.load(current_fold['test']['path']).df
            self._predict_targets = pd.concat((self._valid_targets, self._test_targets))
        else:
            self._predict_targets = self._valid_targets

        labeled_amount = self.train_conf.get('labeled_amount', None)
        if labeled_amount is not None:
            _len_orig = len(self._train_targets)
            if type(labeled_amount) is float:
                labeled_amount = int(_len_orig * labeled_amount)
            self._train_targets = self._train_targets.iloc[:labeled_amount]
            logger.info(f'Reduced train amount from {_len_orig} to {len(self._train_targets)}')

    def build_iterable_processing(self, part):
        yield FeatureTypeCast({self.col_id: self.col_id_dtype})

        if 'dataset_files' in self.setup_conf and self.setup_conf.split_by == 'embeddings_validation':
            if part == 'train':
                yield IdFilter(id_col=self.col_id, relevant_ids=self._train_targets[self.col_id].values.tolist())
            elif part == 'valid':
                yield IdFilter(id_col=self.col_id, relevant_ids=self._valid_targets[self.col_id].values.tolist())
            elif part == 'test':
                yield IdFilter(id_col=self.col_id, relevant_ids=self._test_targets[self.col_id].values.tolist())
            elif part == 'predict':
                yield IdFilter(id_col=self.col_id, relevant_ids=self._predict_targets[self.col_id].values.tolist())
            else:
                raise AttributeError(f'Unknown part: {part}')

        if part == 'train':
            yield SeqLenFilter(min_seq_len=self.train_conf.min_seq_len)

        if part == 'train':
            yield TargetJoin(self.col_id, self._train_targets.set_index(self.col_id)[self.col_target].to_dict(), self.y_function)
        elif part == 'valid':
            yield TargetJoin(self.col_id, self._valid_targets.set_index(self.col_id)[self.col_target].to_dict(), self.y_function)
        elif part == 'test':
            yield TargetJoin(self.col_id, self._test_targets.set_index(self.col_id)[self.col_target].to_dict(), self.y_function)
        elif part == 'predict':
            yield TargetExtractor(self.col_id, drop_from_features=False)
        else:
            raise AttributeError(f'Unknown part: {part}')

        yield FeatureFilter(keep_feature_names=self.category_names,
                            drop_feature_names=self.setup_conf.get('drop_feature_names', None))

        if self._type == 'iterable':
            # all processing in single chain
            if part == 'train':
                yield IterableShuffle(self.train_conf.buffer_size)
            yield IterableAugmentations(self.build_augmentations(part))

    def build_augmentations(self, part):
        if part == 'train':
            return build_augmentations(self.train_conf.augmentations)
        elif part == 'valid':
            return build_augmentations(self.valid_conf.augmentations)
        elif part in ('test', 'predict'):
            return build_augmentations(self.test_conf.augmentations)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=padded_collate_distribution_target if self.distribution_target_task else padded_collate,
            shuffle=False if self._type == 'iterable' else True,
            num_workers=self.train_conf.num_workers,
            batch_size=self.train_conf.batch_size,
            drop_last=self.train_conf.get('drop_last', False)
        )

    def setup_map(self):
        if self.train_dataset is not None:
            self.train_dataset = list(tqdm(iter(self.train_dataset)))
            logger.info(f'Loaded {len(self.train_dataset)} for train')
            self.train_dataset = MapAugmentationDataset(
                base_dataset=self.train_dataset,
                a_chain=self.build_augmentations('train'),
            )
        if self.valid_dataset is not None:
            self.valid_dataset = list(tqdm(iter(self.valid_dataset)))
            logger.info(f'Loaded {len(self.valid_dataset)} for valid')
            self.valid_dataset = MapAugmentationDataset(
                base_dataset=self.valid_dataset,
                a_chain=self.build_augmentations('valid'),
            )
        if self.test_dataset is not None:
            self.test_dataset = list(tqdm(iter(self.test_dataset)))
            logger.info(f'Loaded {len(self.test_dataset)} for test')
            self.test_dataset = MapAugmentationDataset(
                base_dataset=self.test_dataset,
                a_chain=self.build_augmentations('test'),
            )
        if self.predict_dataset is not None:
            self.predict_dataset = list(tqdm(iter(self.predict_dataset)))
            logger.info(f'Loaded {len(self.predict_dataset)} for predict')
            self.predict_dataset = MapAugmentationDataset(
                base_dataset=self.predict_dataset,
                a_chain=self.build_augmentations('predict'),
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            collate_fn=padded_collate_distribution_target if self.distribution_target_task else padded_collate,
            num_workers=self.valid_conf.num_workers,
            batch_size=self.valid_conf.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=padded_collate_distribution_target if self.distribution_target_task else padded_collate,
            num_workers=self.test_conf.num_workers,
            batch_size=self.test_conf.batch_size,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_dataset,
            collate_fn=padded_collate,
            num_workers=self.test_conf.num_workers,
            batch_size=self.test_conf.batch_size,
        )
