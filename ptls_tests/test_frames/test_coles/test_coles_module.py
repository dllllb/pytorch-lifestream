import pytorch_lightning as pl
import torch.optim

from ptls.frames.coles import CoLESModule
from ptls_tests.test_data_load import RandomEventData
from pyhocon import ConfigFactory
from ptls.nn.seq_encoder import RnnSeqEncoder
from ptls.nn import Head
from ptls.nn import TrxEncoder
from functools import partial


def tst_params():
    params = {
        "data_module": {
            "train": {
                "num_workers": 1,
                "batch_size": 32,
                "trx_dropout": 0.01,
                "max_seq_len": 100,
            },
            "valid": {
                "batch_size": 16,
                "num_workers": 1,
                "max_seq_len": 100
            }
        },
        "rnn": {
            "type": "gru",
            "hidden_size": 16,
            "bidir": False,
            "trainable_starter": "static"
        },
        "trx_encoder": {
            "embeddings_noise": .003,
            "norm_embeddings": False,
            'embeddings': {
                'mcc_code': {'in': 21, 'out': 3},
                'trans_type': {'in': 11, 'out': 2},
            },
            'numeric_values': {'amount': 'log'}
        },
    }

    params = ConfigFactory.from_dict(params)
    return params


def test_train_loop():
    params = tst_params()

    model = CoLESModule(
        seq_encoder=RnnSeqEncoder(
            trx_encoder=TrxEncoder(**params['trx_encoder']),
            **params['rnn'],
        ),
        head=Head(use_norm_encoder=True),
        optimizer_partial=partial(torch.optim.Adam),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=1.0),
    )
    dl = RandomEventData(params['data_module'])
    trainer = pl.Trainer(max_epochs=1, logger=None, checkpoint_callback=False)
    trainer.fit(model, dl)
