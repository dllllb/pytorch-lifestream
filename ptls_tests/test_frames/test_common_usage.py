"""Test examples from docs/frames/common_usage.md
"""
from functools import partial

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split

from ptls.data_load.datasets.dataloaders import inference_data_loader
from ptls.frames.coles import CoLESModule
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames.supervised import SequenceToTarget
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames import PtlsDataModule


def test_dataset_way():
    # Makes 1000 samples with `mcc_code` and `amount` features and seq_len randomly sampled in range (100, 200)
    dataset = [{
        'mcc_code': torch.randint(1, 10, (seq_len,)),
        'amount': torch.randn(seq_len),
        'event_time': torch.arange(seq_len),
    } for seq_len in torch.randint(100, 200, (1000,))]

    # split 10% for validation
    train_data, valid_data = train_test_split(dataset, test_size=0.1)

    # datasets
    splitter=SampleSlices(split_count=5, cnt_min=10, cnt_max=20)
    train_dataset = ColesDataset(data=train_data, splitter=splitter)
    valid_dataset = ColesDataset(data=valid_data, splitter=splitter)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=4,
        batch_size=32,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        collate_fn=valid_dataset.collate_fn,
        shuffle=False,
        num_workers=0,
        batch_size=32,
    )

    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoder(
            embeddings={'mcc_code': {'in': 10, 'out': 4}},
            numeric_values={'amount': 'identity'},
        ),
        hidden_size=16,  # this is final embedding size
    )

    coles_module = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=0.001),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9),
    )

    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(coles_module, train_dataloader, valid_dataloader)

    # inference
    inference_dataloader = inference_data_loader(dataset, num_workers=4, batch_size=256)
    model = SequenceToTarget(seq_encoder)
    trainer = pl.Trainer()
    embeddings = torch.vstack(trainer.predict(model, inference_dataloader))
    assert embeddings.size() == (1000, 16)


def test_datamodule_way():
    # Makes 1000 samples with `mcc_code` and `amount` features and seq_len randomly sampled in range (100, 200)
    dataset = [{
        'mcc_code': torch.randint(1, 10, (seq_len,)),
        'amount': torch.randn(seq_len),
        'event_time': torch.arange(seq_len),
    } for seq_len in torch.randint(100, 200, (1000,))]

    # split 10% for validation
    train_data, valid_data = train_test_split(dataset, test_size=0.1)

    # datasets
    splitter=SampleSlices(split_count=5, cnt_min=10, cnt_max=20)
    train_dataset = ColesDataset(data=train_data, splitter=splitter)
    valid_dataset = ColesDataset(data=valid_data, splitter=splitter)

    datamodule = PtlsDataModule(
        train_data=train_dataset,
        train_batch_size=32,
        train_num_workers=4,
        valid_data=valid_dataset,
        valid_num_workers=0,
    )

    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoder(
            embeddings={'mcc_code': {'in': 10, 'out': 4}},
            numeric_values={'amount': 'identity'},
        ),
        hidden_size=16,  # this is final embedding size
    )

    coles_module = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=0.001),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9),
    )

    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(coles_module, datamodule)

    # inference
    inference_dataloader = inference_data_loader(dataset, num_workers=4, batch_size=256)
    model = SequenceToTarget(seq_encoder)
    trainer = pl.Trainer()
    embeddings = torch.vstack(trainer.predict(model, inference_dataloader))
    assert embeddings.size() == (1000, 16)
