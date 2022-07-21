from functools import partial
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split

from ptls.data_load.data_module.emb_data_module import train_data_loader
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.datasets.dataloaders import inference_data_loader
from ptls.data_load.iterable_processing import ISeqLenLimit, SeqLenFilter
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.preprocessing.pandas_preprocessor import PandasDataPreprocessor
from ptls.frames.coles import CoLESModule, ColesDataset
from ptls.nn.seq_encoder import RnnSeqEncoder
from ptls.nn.trx_encoder import TrxEncoder


def test_train_inference():
    source_data = pd.read_csv(Path(__file__).parent / "age-transactions.csv")

    preprocessor = PandasDataPreprocessor(
        col_id='client_id',
        col_event_time='trans_date',
        event_time_transformation='none',
        cols_category=["trans_date", "small_group"],
        cols_numerical=["amount_rur"],
    )

    dataset = preprocessor.fit_transform(source_data)
    train, test = train_test_split(dataset, test_size=0.4, random_state=42)
    train_ds = ColesDataset(
        data=MemoryMapDataset(data=train, i_filters=[SeqLenFilter(min_seq_len=5)]),
        splitter=SampleSlices(
            split_count=3,
            cnt_min=3,
            cnt_max=20,
        ),
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        collate_fn=train_ds.collate_fn,
        shuffle=True,
        num_workers=0,
        batch_size=4
    )

    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoder(
            embeddings={k: {'in': v, 'out': 16} for k, v in preprocessor.get_category_dictionary_sizes().items()},
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
    trainer.fit(model, train_dl)

    test_dl = inference_data_loader(test, num_workers=0, batch_size=4)

    trainer.predict(model, test_dl)
