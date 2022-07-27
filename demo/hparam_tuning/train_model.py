from functools import partial

import hydra
import os
import pytorch_lightning as pl
import torch
from ptls.data_load.datasets import ParquetDataset
from torchmetrics import Accuracy, AUROC

from data_preprocessing import get_file_name_train, get_file_name_test
from ptls.data_load.datasets import PersistDataset, parquet_file_scan
from ptls.data_load.iterable_processing import TargetEmptyFilter

from ptls.frames import PtlsDataModule
from ptls.frames.supervised import SequenceToTarget, SeqToTargetDataset
from ptls.loss import BCELoss
from ptls.nn import RnnSeqEncoder, TrxEncoder

import logging


logger = logging.getLogger(__name__)



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    train_data = PersistDataset(
        ParquetDataset(
            parquet_file_scan(os.path.join(conf.preprocessing.folds_path, get_file_name_train(conf.fold_id))),
            i_filters=[
                TargetEmptyFilter(target_col='target_gender'),
            ],
        )
    )
    test_data = PersistDataset(
        ParquetDataset(
            parquet_file_scan(os.path.join(conf.preprocessing.folds_path, get_file_name_test(conf.fold_id))),
        ),
    )

    data_module = PtlsDataModule(
        train_data=SeqToTargetDataset(train_data, target_col_name='target_gender', target_dtype='int'),
        test_data=SeqToTargetDataset(test_data, target_col_name='target_gender', target_dtype='int'),
        train_batch_size=32,
        valid_batch_size=256,
        train_num_workers=8,
        valid_num_workers=8,
        train_drop_last=True,
    )

    pl_module = SequenceToTarget(
        seq_encoder=RnnSeqEncoder(
            trx_encoder=TrxEncoder(
                embeddings={
                    'mcc_code': {'in': 180, 'out': 32},
                    'tr_type': {'in': 80, 'out': 8},
                },
                numeric_values={'amount': 'log'},
            ),
            hidden_size=64,
        ),
        head=torch.nn.Sequential(
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
            torch.nn.Flatten(0),
        ),
        loss=BCELoss(),
        metric_list={
            'acc': Accuracy(),
            'auroc': AUROC(),
        },
        optimizer_partial=partial(torch.optim.Adam),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=1000, gamma=1.0),
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=3,
        checkpoint_callback=False,
    )
    trainer.fit(pl_module, data_module)
    logger.info(f'logged_metrics={trainer.logged_metrics}')

    test_metrics = trainer.test(pl_module, data_module, verbose=False)

    logger.info(f'logged_metrics={trainer.logged_metrics}')
    logger.info(f'test_metrics={test_metrics}')

    final_test_metric = test_metrics[0]['test_auroc']
    return final_test_metric

if __name__ == '__main__':
    main()
