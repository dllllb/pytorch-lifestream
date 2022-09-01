"""
Inspired by:
https://medium.com/optuna/easy-hyperparameter-management-with-hydra-mlflow-and-optuna-783730700e7d


# Data
Consider labeled and unlabeled data.
Labeled data splits into N folds, where `N = N_valid + N_test`.
`N_valid` - is count of folds used to measure a quality during hparam optimisation.
`N_test` - is count of folds used to measure a final quality
Unlabeled data used (if needed) in each fold split.

# Main algorithm
>>> def estimate_model(hparams, folds):
>>>     metric = []
>>>     for ds_train, ds_valid in folds:
>>>         model = Model(hparams)
>>>         model.fit(ds_train)
>>>         metric_on_fold = model.test(ds_valid)
>>>         metric.append(metric_on_fold)
>>>     return metric.mean()

Hydra multirun with optuna framework choose a next hparams and run `estimate_model` on validation folds.

# Supported features with hydra
- Model pretrain. Should be a part of `estimate_model` function.
- LGBM downstream model supported. Should be a part of `estimate_model` function.
- Tensorboard full logging:
    - Each fold logged as usual
    - Mean metrics logged with hparams and hydra.cwd to link tb metrics and hydra outputs.
    - Hydra outputs logs a tb versions to link runs, configs and results.
- Multiprocess parallel run. With hydra launcher customisation.

# Features not supported by hydra
- Each pretrain epoch yield a model tor finetuning.
    Workaround: Returns finetuned results from best epoch.
- Fast hparam selection with early epoch estimation and break low quality configurations.
    Workaround: Early stop for worst configurations.
    Track previous results from tb and break a runs that a worse than previous.

"""


import logging
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy, AUROC

from data_preprocessing import get_file_name_train, get_file_name_test
from ptls.data_load import iterable_processing
from ptls.data_load.datasets import ParquetDataset
from ptls.data_load.datasets import PersistDataset, parquet_file_scan
from ptls.frames import PtlsDataModule
from ptls.frames.supervised import SequenceToTarget, SeqToTargetDataset
from ptls.loss import BCELoss
from ptls.nn import RnnSeqEncoder, TrxEncoder

logger = logging.getLogger(__name__)


def flat_conf(conf):
    def _explore():
        for param_name, element in conf.items():
            for k, v in _explore_recursive(param_name, element):
                yield k, v
        yield 'hydra.cwd', Path.cwd()

    def _explore_recursive(parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    _explore_recursive(f'{parent_name}.{k}', v)
                else:
                    yield f'{parent_name}.{k}', v
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                yield f'{parent_name}.{i}', v
        else:
            yield parent_name, element

    return dict(_explore())


def get_data_module(conf, fold_id):
    folds_path = Path(conf.preprocessing.folds_path)
    train_files = parquet_file_scan(to_absolute_path(folds_path / get_file_name_train(fold_id)))

    train_files, valid_files = train_test_split(train_files, test_size=conf.validation_rate)
    train_data = PersistDataset(
        ParquetDataset(
            train_files,
            i_filters=[
                iterable_processing.TargetEmptyFilter(target_col='target_gender'),
                iterable_processing.ISeqLenLimit(max_seq_len=2000),
            ],
        )
    )
    valid_data = PersistDataset(
        ParquetDataset(
            valid_files,
            i_filters=[
                iterable_processing.TargetEmptyFilter(target_col='target_gender'),
                iterable_processing.ISeqLenLimit(max_seq_len=2000),
            ],
        )
    )
    test_data = PersistDataset(
        ParquetDataset(
            parquet_file_scan(to_absolute_path(folds_path / get_file_name_test(fold_id))),
            i_filters=[
                iterable_processing.ISeqLenLimit(max_seq_len=2000),
            ],
        ),
    )
    data_module = PtlsDataModule(
        train_data=SeqToTargetDataset(train_data, target_col_name='target_gender', target_dtype='int'),
        valid_data=SeqToTargetDataset(valid_data, target_col_name='target_gender', target_dtype='int'),
        test_data=SeqToTargetDataset(test_data, target_col_name='target_gender', target_dtype='int'),
        train_batch_size=32,
        valid_batch_size=256,
        train_num_workers=8,
        valid_num_workers=8,
        train_drop_last=True,
    )
    return data_module


def get_pl_module(conf):
    head_layers = []
    if conf.pl_module.batch_norm1:
        head_layers.append(torch.nn.BatchNorm1d(conf.pl_module.hidden_size1))
    if conf.pl_module.relu1:
        head_layers.append(torch.nn.ReLU())
    if conf.pl_module.dropout1 > 0:
        head_layers.append(torch.nn.Dropout(conf.pl_module.dropout1))
    if conf.pl_module.num_layers == 1:
        head_layers.append(torch.nn.Linear(conf.pl_module.hidden_size1, 1))
    elif conf.pl_module.num_layers == 2:
        head_layers.append(torch.nn.Linear(conf.pl_module.hidden_size1, conf.pl_module.hidden_size2))
        if conf.pl_module.batch_norm2:
            head_layers.append(torch.nn.BatchNorm1d(conf.pl_module.hidden_size2))
        if conf.pl_module.relu2:
            head_layers.append(torch.nn.ReLU())
        if conf.pl_module.dropout2 > 0:
            head_layers.append(torch.nn.Dropout(conf.pl_module.dropout2))
        head_layers.append(torch.nn.Linear(conf.pl_module.hidden_size2, 1))

    head_layers.extend([
        torch.nn.Sigmoid(),
        torch.nn.Flatten(0),
    ])

    pl_module = SequenceToTarget(
        seq_encoder=RnnSeqEncoder(
            trx_encoder=TrxEncoder(
                embeddings={
                    'mcc_code': {'in': 180, 'out': 32},
                    'tr_type': {'in': 80, 'out': 8},
                },
                numeric_values={'amount': 'log'},
            ),
            hidden_size=conf.pl_module.hidden_size1,
        ),
        head=torch.nn.Sequential(*head_layers),
        loss=BCELoss(),
        metric_list={
            'auroc': AUROC(),
        },
        optimizer_partial=partial(torch.optim.Adam, lr=conf.pl_module.lr),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=conf.pl_module.lr_gamma),
    )
    return pl_module


def model_run(conf, fold_id):
    logger.info(f'Start with fold_id={fold_id}')

    data_module = get_data_module(conf, fold_id)

    pl_module = get_pl_module(conf)
    trainer = pl.Trainer(
        gpus=1,
        limit_train_batches=conf.trainer.limit_train_batches,
        max_epochs=conf.trainer.max_epochs,
        enable_checkpointing=False,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=to_absolute_path(conf.tb_save_dir),
            name=f'{conf.mode}_fold={fold_id}',
            version=None,
            default_hp_metric=False,
        ),
    )
    trainer.fit(pl_module, data_module)
    logger.info(f'logged_metrics={trainer.logged_metrics}')
    test_metrics = trainer.test(pl_module, data_module, verbose=False)
    logger.info(f'logged_metrics={trainer.logged_metrics}')
    logger.info(f'test_metrics={test_metrics}')
    final_test_metric = test_metrics[0]['test_auroc']

    return final_test_metric


def get_fold_list(conf):
    if conf.mode == 'valid':
        fold_list = [i for i in range(conf.preprocessing.fold_count_valid)]
    elif conf.mode == 'test':
        fold_list = [i + conf.preprocessing.fold_count_valid for i in range(conf.preprocessing.fold_count_test)]
    else:
        raise AttributeError(f'Mode can be `valid` or `test`. Found: {conf.mode}')
    return fold_list


def log_resuts(conf, fold_list, results, float_precision='{:.4f}'):
    mean = np.mean(results)
    std = np.std(results)

    results_str = ', '.join([float_precision.format(r) for r in results])
    logger.info(', '.join([
        f'{conf.mode} done',
        f'folds={fold_list}',
        f'mean={float_precision.format(mean)}',
        f'std={float_precision.format(std)}',
        f'interval_pm=[{float_precision.format(mean - std)}, {float_precision.format(mean + std)}]',
        f'values=[{results_str}]',
    ]))

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=to_absolute_path(conf.tb_save_dir),
        name=f'{conf.mode}_mean',
        version=None,
        prefix='',
        default_hp_metric=False,
    )
    tb_logger.log_hyperparams(
        params=flat_conf(conf),
        metrics={f'auroc_mean': mean},
    )
    logger.info(f'Results are logged to tensorboard as {tb_logger.name}/{tb_logger.version}')


@hydra.main(version_base=None, config_path="conf")
def main(conf):
    fold_list = get_fold_list(conf)

    results = []
    for fold_id in fold_list:
        result_on_fold = model_run(conf, fold_id)
        logger.info(f'{conf.mode}, fold={fold_id}, metric={result_on_fold:.6f}')
        results.append(result_on_fold)
    results_mean = np.mean(results)

    log_resuts(conf, fold_list, results)

    return results_mean


if __name__ == '__main__':
    main()
