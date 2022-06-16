from functools import partial
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split

from ptls.data_load.data_module.emb_data_module import train_data_loader, inference_data_loader
from ptls.data_preprocessing.pandas_preprocessor import PandasDataPreprocessor
from ptls.frames.coles import CoLESModule
from ptls.seq_encoder import RnnSeqEncoder
from ptls.trx_encoder import TrxEncoder


def test_train_inference():
    source_data = pd.read_csv(Path(__file__).parent / "age-transactions.csv")

    preprocessor = PandasDataPreprocessor(
        col_id='client_id',
        cols_event_time='trans_date',
        time_transformation='float',
        cols_category=["trans_date", "small_group"],
        cols_log_norm=["amount_rur"],
        cols_identity=[],
        print_dataset_info=False,
    )

    dataset = preprocessor.fit_transform(source_data)

    train, test = train_test_split(dataset, test_size=0.4, random_state=42)

    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoder(
            embeddings={k: {'in': v, 'out': 16} for k, v in preprocessor.get_category_sizes().items()},
            numeric_values={},
            embeddings_noise=0.003,
        ),
        hidden_size=16,
        type='gru',
    )

    model = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=1.0),
    )

    trainer = pl.Trainer(
        max_epochs=1,
        gpus=0 if torch.cuda.is_available() else 0,
        logger=False
    )

    train_dl = train_data_loader(
        train,
        min_seq_len=5,
        seq_split_strategy='SampleSlices',
        split_count=3,
        split_cnt_min=3,
        split_cnt_max=20,
        num_workers=0,
        batch_size=4
    )

    trainer.fit(model, train_dl)

    test_dl = inference_data_loader(test, num_workers=0, batch_size=4)

    trainer.predict(model, test_dl)
