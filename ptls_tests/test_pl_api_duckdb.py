from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch

from ptls.data_load import padded_collate_wo_target
from ptls.data_load.datasets import DuckDbDataset
from ptls.data_load.iterable_processing import FilterNonArray, ISeqLenLimit
from ptls.frames.coles import CoLESModule, ColesIterableDataset
from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.nn.seq_encoder import RnnSeqEncoder
from ptls.nn.trx_encoder import TrxEncoder


def test_train_inference():
    train_data = f"""
        (SELECT * FROM read_csv_auto('{Path(__file__).parent / 'age-transactions.csv'}')
        WHERE hash(client_id) % 5 <> 0)
        """

    train_ds = DuckDbDataset(
        data_read_func=train_data,
        col_id='client_id',
        col_event_time='trans_date',
        col_event_fields=['amount_rur', 'small_group']
    )

    c_train_ds = ColesIterableDataset(
        data=train_ds,
        splitter=SampleSlices(
            split_count=3,
            cnt_min=3,
            cnt_max=20,
        ),
        col_time='trans_date'
    )

    train_dl = torch.utils.data.DataLoader(
        c_train_ds,
        collate_fn=c_train_ds.collate_fn,
        num_workers=0,
        batch_size=4
    )

    cat_sizes = train_ds.get_category_sizes(['small_group'])

    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoder(
            embeddings={
                k: {'in': v, 'out': 4}
                for k, v in cat_sizes.items()},
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
        loss=ContrastiveLoss(margin=0.5, sampling_strategy=HardNegativePairSelector(2))
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        logger=False
    )
    trainer.fit(model, train_dl)

    test_data = f"""
        (SELECT * FROM read_csv_auto('{Path(__file__).parent / 'age-transactions.csv'}')
        WHERE hash(client_id) % 5 == 0)
        """

    test_ds = DuckDbDataset(
        data_read_func=test_data,
        col_id='client_id',
        col_event_time='trans_date',
        col_event_fields=['amount_rur', 'small_group'],
        i_filters=[
            FilterNonArray(),
            ISeqLenLimit(max_seq_len=100),
        ]
    )

    test_dl = torch.utils.data.DataLoader(
        dataset=test_ds,
        collate_fn=padded_collate_wo_target,
        shuffle=False,
        num_workers=0,
        batch_size=4,
    )

    trainer.predict(model, test_dl)
