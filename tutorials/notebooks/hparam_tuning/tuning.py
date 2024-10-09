"""
Inspired by:
https://medium.com/optuna/easy-hyperparameter-management-with-hydra-mlflow-and-optuna-783730700e7d

# Supported features with hydra
- Model pretrain. Should be a part of `model_run` function.
- LGBM downstream model supported. Should be a part of `model_run` function.
- Tensorboard full logging:
    - Each fold logged as usual
    - Mean metrics logged with hparams, hydra.cwd and hydra.reuse_cmd to link tb metrics and hydra outputs.
    - Hydra outputs logs a tb versions to link runs, configs and results.
- Multiprocess parallel run. With hydra launcher customization.

# Features not supported by hydra
- Each pretrain epoch yield a model tor finetuning.
    Workaround: Returns finetuned results from best epoch.
- Fast hparam selection with early epoch estimation and break low quality configurations (hyperband).
    Workaround: Early stop for worst configurations.
    Track previous results from tb and break a runs that a worse than previous.

"""


import logging
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import scipy.stats
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torchmetrics import AUROC

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
        yield 'hydra.reuse_cmd', f'--config-dir={Path.cwd()} +conf_override@=config'

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
    folds_path = Path(conf.data_preprocessing.folds_path)
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
    head_layers.append(torch.nn.Linear(conf.pl_module.hidden_size1, 1))

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
    train_auroc_t = trainer.logged_metrics['train_auroc'].item()
    train_auroc_v = trainer.logged_metrics['val_auroc'].item()
    test_metrics = trainer.test(pl_module, data_module, verbose=False)
    logger.info(f'logged_metrics={trainer.logged_metrics}')
    logger.info(f'test_metrics={test_metrics}')
    final_test_metric = test_metrics[0]['test_auroc']

    if conf.mode == 'valid':
        trainer.logger.log_hyperparams(
            params=flat_conf(conf),
            metrics={
                f'hp/auroc': final_test_metric,
                f'hp/auroc_t': train_auroc_t,
                f'hp/auroc_v': train_auroc_v,
            },
        )

    logger.info(f'[{conf.mode}] on fold[{fold_id}] finished with {final_test_metric:.4f}')
    return final_test_metric


def log_resuts(conf, fold_list, results, float_precision='{:.4f}'):
    def t_interval(x, p=0.95):
        eps = 1e-9
        n = len(x)
        s = x.std(ddof=1)

        return scipy.stats.t.interval(p, n - 1, loc=x.mean(), scale=(s + eps) / (n ** 0.5))

    mean = results.mean()
    std = results.std()
    t_int = t_interval(results)

    results_str = ', '.join([float_precision.format(r) for r in results])
    logger.info(', '.join([
        f'{conf.mode} done',
        f'folds={fold_list}',
        f'mean={float_precision.format(mean)}',
        f'std={float_precision.format(std)}',
        f'mean_pm_std=[{float_precision.format(mean - std)}, {float_precision.format(mean + std)}]',
        f'confidence95=[{float_precision.format(t_int[0])}, {float_precision.format(t_int[1])}]',
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
        metrics={f'{conf.mode}_auroc_mean': mean},
    )
    logger.info(f'Results are logged to tensorboard as {tb_logger.name}/{tb_logger.version}')
    logger.info(f'Output logged to "{Path.cwd()}"')


def main_valid(conf):
    valid_fold = 0
    result_fold = model_run(conf, valid_fold)
    logger.info('Validation done')
    return result_fold


def main_test(conf):
    test_folds = [i for i in range(1, conf.data_preprocessing.n_folds)]
    results = []
    for fold_id in test_folds:
        result_fold = model_run(conf, fold_id)
        results.append(result_fold)
    results = np.array(results)

    log_resuts(conf, test_folds, results)

    return results.mean()


@hydra.main(version_base=None, config_path="conf")
def main(conf):
    # save config for future overrides
    conf_override_path = Path.cwd() / 'conf_override'
    conf_override_path.mkdir()
    OmegaConf.save(config=conf, f=conf_override_path / 'config.yaml')

    if conf.mode == 'valid':
         return main_valid(conf)
    if conf.mode == 'test':
         return main_test(conf)
    raise AttributeError(f'`conf.mode should be valid or test. Found: {conf.mode}')


if __name__ == '__main__':
    main()
