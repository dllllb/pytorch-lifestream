import time
from functools import partial

import pytorch_lightning as pl
import torch
from pyinstrument import Profiler
from pytorch_lightning.strategies import DDPStrategy
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR

from ptls.data_load.augmentations import DropoutTrx
from ptls.data_load.datasets import parquet_file_scan
from ptls.data_load.datasets.augmentation_dataset import AugmentationIterableDataset
from ptls.data_load.datasets.parquet_dataset import (
    DistributedParquetDataset,
    ParquetDataset,
)
from ptls.data_load.iterable_processing import (
    FeatureFilter,
    IterableShuffle,
    SeqLenFilter,
)
from ptls.frames import PtlsDataModule
from ptls.frames.coles import ColesIterableDataset, CoLESModule
from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.nn import L2NormEncoder, RnnSeqEncoder, TrxEncoder

data_path = '../data'

SEED = 42
EMBEDDINGS = {
    "currency": {"in": 13, "out": 2},
    "operation_kind": {"in": 9, "out": 2},
    "card_type": {"in": 177, "out": 0},
    "operation_type": {"in": 24, "out": 4},
    "operation_type_group": {"in": 6, "out": 32},
    "ecommerce_flag": {"in": 5, "out": 1},
    "payment_system": {"in": 9, "out": 4},
    "income_flag": {"in": 5, "out": 1},
    "mcc": {"in": 110, "out": 32},
    "country": {"in": 26, "out": 0},
    "city": {"in": 163, "out": 0},
    "mcc_category": {"in": 30, "out": 16},
    "day_of_week": {"in": 9, "out": 2},
    "hour": {"in": 25, "out": 4},
    "weekofyear": {"in": 55, "out": 4},
}


if __name__ == "__main__":

    pl.seed_everything(SEED)

    profiler = Profiler()

    profiler.start()

    model = CoLESModule(
        validation_metric=BatchRecallTopK(K=4, metric="cosine"),
        seq_encoder=RnnSeqEncoder(
            trx_encoder=TrxEncoder(
                norm_embeddings=False,
                embeddings_noise=0.003,
                embeddings=EMBEDDINGS,
                numeric_values={"amnt": "identity", "hour_diff": "log"},
            ),
            type="gru",
            hidden_size=1024,
            bidir=False,
            trainable_starter="static",
        ),
        head=L2NormEncoder(),
        loss=ContrastiveLoss(
            margin=0.5, 
            sampling_strategy=HardNegativePairSelector(neg_count=5),
            distributed_mode=True
        ),
        optimizer_partial=partial(Adam, **{"lr": 0.001, "weight_decay": 0.0}),
        lr_scheduler_partial=partial(StepLR, **{"step_size": 1, "gamma": 0.8}),
    )

    dm = PtlsDataModule(
        train_data=ColesIterableDataset(
            splitter=SampleSlices(split_count=5, cnt_min=20, cnt_max=60),
            data=AugmentationIterableDataset(
                f_augmentations=[DropoutTrx(trx_dropout=0.01)],
                # data=DistributedParquetDataset(
                data=ParquetDataset(
                    shuffle_files=True,
                    i_filters=[
                        SeqLenFilter(max_seq_len=30),
                        FeatureFilter(drop_feature_names=["app_id", "product", "flag"]),
                        IterableShuffle(buffer_size=10000),
                    ],
                    data_files=parquet_file_scan(
                        file_path=f"{data_path}/train_trx.parquet",
                        valid_rate=0.05,
                        return_part="train",
                    ),
                    # max_items_per_file=130000,
                ),
            ),
        ),
        valid_data=ColesIterableDataset(
            splitter=SampleSlices(split_count=5, cnt_min=20, cnt_max=60),
            # data=DistributedParquetDataset(
            data=ParquetDataset(
                i_filters=[
                    SeqLenFilter(min_seq_len=30),
                    FeatureFilter(drop_feature_names=["app_id", "product", "flag"]),
                ],
                data_files=parquet_file_scan(
                    file_path=f"{data_path}/train_trx.parquet",
                    valid_rate=0.05,
                    return_part="valid",
                ),
                # max_items_per_file=100,
            ),
        ),
        train_batch_size=128,
        # train_batch_size=256,
        train_num_workers=64,
        # valid_batch_size=256,
        valid_batch_size=128,
        valid_num_workers=64,
    )
    strategy = DDPStrategy(find_unused_parameters=False)

    torch.set_float32_matmul_precision("medium")

    start = time.time()

    trainer = pl.Trainer(
        max_epochs=1,
        deterministic=True,
        check_val_every_n_epoch=1,
        devices=2,
        strategy=strategy,
    )
    trainer.fit(model, dm)
    
    end = time.time()

    print(40 * "*")
    print(f"-----------Elapsed Time: {end-start} sec-----------")
    print(40 * "*")

    profiler.stop()
    profiler.print()
