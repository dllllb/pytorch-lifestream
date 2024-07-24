import os
import pandas as pd
import torch
import ptls
from ptls.data_load.datasets import ParquetFiles, ParquetDataset, MemoryMapDataset
from ptls.frames import PtlsDataModule
from ptls.frames.coles import ColesDataset, ColesIterableDataset
from ptls.frames.supervised import SeqToTargetIterableDataset
from sklearn.model_selection import train_test_split
from ptls.frames.coles.split_strategy import SampleSlices


def get_age_pred_coles_datamodule(fold_i, **kwargs):
    df_trx_pretrain = pd.read_pickle(f'idx_data/fold_{fold_i}/df_trx_pretrain.pickle')
    df_seq_pretrain = pd.read_pickle(f'idx_data/fold_{fold_i}/df_seq_pretrain.pickle')

    df_seq_pretrain_train, df_seq_pretrain_valid = train_test_split(
        df_seq_pretrain, test_size=0.05, shuffle=True, random_state=42)

    coles_datamodule = PtlsDataModule(
        train_data=ColesDataset(
            data=MemoryMapDataset(
                df_seq_pretrain_train.to_dict(orient='records') +
                df_trx_pretrain.to_dict(orient='records')
            ),
            splitter=SampleSlices(
                split_count=5,
                cnt_min=25,
                cnt_max=200,
            ),
        ),
        valid_data=ColesDataset(
            data=MemoryMapDataset(
                df_seq_pretrain_train.to_dict(orient='records')),
            splitter=SampleSlices(
                split_count=5,
                cnt_min=25,
                cnt_max=100,
            ),
        ),
        train_batch_size=64,
        train_num_workers=4,
        valid_batch_size=512,
        valid_num_workers=4,
    )

    return coles_datamodule


def get_synthetic_coles_datamodule(path, **kwargs):
    train_files = ParquetFiles(os.path.join(path, "train"))
    train_dataset = ParquetDataset(train_files, shuffle_files=True)
    eval_files = ParquetFiles(os.path.join(path, "eval"))
    eval_dataset = ParquetDataset(eval_files, shuffle_files=True)

    coles_datamodule = PtlsDataModule(
        train_data=ColesIterableDataset(
            train_dataset,
            splitter=SampleSlices(
                split_count=5,
                cnt_min=70,
                cnt_max=100,
            ),
        ),
        valid_data=ColesIterableDataset(
            eval_dataset,
            splitter=SampleSlices(
                split_count=5,
                cnt_min=70,
                cnt_max=100, ),
        ),
        train_num_workers=4,
        train_batch_size=512,
        valid_num_workers=4,
        valid_batch_size=512,
    )

    return coles_datamodule


def get_age_pred_sup_datamodule(fold_i, **kwargs):
    df_gbm_train = pd.read_pickle(f'idx_data/fold_{fold_i}/df_gbm_train.pickle')
    df_gbm_test = pd.read_pickle(f'idx_data/fold_{fold_i}/df_gbm_test.pickle')

    test_dataset = ptls.data_load.datasets.MemoryMapDataset(
        df_gbm_test.to_dict(orient='records'),
        i_filters=[
            ptls.data_load.iterable_processing.ISeqLenLimit(max_seq_len=2000),
        ],
    )

    train_dataset = ptls.data_load.datasets.MemoryMapDataset(
        df_gbm_train.to_dict(orient='records'),
        i_filters=[
            ptls.data_load.iterable_processing.ISeqLenLimit(max_seq_len=2000),
        ],
    )

    sup_datamodule = PtlsDataModule(
        train_data=SeqToTargetIterableDataset(train_dataset, target_col_name='bins', target_dtype=torch.long),
        test_data=SeqToTargetIterableDataset(test_dataset, target_col_name='bins', target_dtype=torch.long),
        train_batch_size=512,
        test_batch_size=512,
        train_num_workers=4,
        test_num_workers=4,
    )
    return sup_datamodule


def get_synthetic_sup_datamodule(fold_i, **kwargs):
    path = "syndata/" + str(fold_i) + "/"

    train_files = ParquetFiles(os.path.join(path, "train"))
    train_dataset = ParquetDataset(train_files, shuffle_files=True)
    eval_files = ParquetFiles(os.path.join(path, "eval"))
    test_dataset = ParquetDataset(eval_files, shuffle_files=True)

    sup_datamodule = PtlsDataModule(
        train_data=SeqToTargetIterableDataset(train_dataset, target_col_name='class_label', target_dtype=torch.long),
        test_data=SeqToTargetIterableDataset(test_dataset, target_col_name='class_label', target_dtype=torch.long),
        train_batch_size=512,
        test_batch_size=512,
        train_num_workers=4,
        test_num_workers=4,
    )
    return sup_datamodule
