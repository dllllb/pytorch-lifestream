import pytorch_lightning as pl
import torch
from pyhocon import ConfigFactory

from ptls.data_load import TrxDataset
from ptls.data_load import create_train_loader, create_validation_loader
from ptls.frames.cpc import CpcModule
from ptls_tests.utils.data_generation import gen_trx_data
from ptls.nn import RnnSeqEncoder
from ptls.nn import TrxEncoder
from functools import partial



def tst_params():
    params = {
        'rnn': {
            'trainable_starter': 'empty',
            'hidden_size': 16,
            'type': 'gru',
            'bidir': False
        },
        "trx_encoder": {
            "embeddings_noise": .1,
            "norm_embeddings": False,
            'embeddings': {
                'mcc_code': {'in': 21, 'out': 3},
                'trans_type': {'in': 11, 'out': 2},
            },
            'numeric_values': {'amount': 'log'}
        },
        "train": {
            "weight_decay": 0,
            "lr": 0.004,
            "batch_size": 32,
            "num_workers": 1,
            "trx_dropout": .1,
            "n_epoch": 1,
            "max_seq_len": 30,
        },
        "valid": {
            "batch_size": 32,
            "num_workers": 1,
            "max_seq_len": 30,
            "recall_top_k": 3
        },
    }

    params = ConfigFactory.from_dict(params)
    return params


def test_rnn_model():
    config = tst_params()

    pl_module = CpcModule(
        seq_encoder=RnnSeqEncoder(
            trx_encoder=TrxEncoder(**config['trx_encoder']),
            **config['rnn'],
        ),
        n_negatives=10, n_forward_steps=3,
        optimizer_partial=partial(torch.optim.Adam),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=1.0),
    )
    train_data = gen_trx_data((torch.rand(1000)*60+1).long())
    train_ds = TrxDataset(train_data)
    test_data = gen_trx_data((torch.rand(100)*60+1).long())
    valid_ds = TrxDataset(test_data)

    train_loader = create_train_loader(train_ds, config['train'])
    valid_loader = create_validation_loader(valid_ds, config['valid'])

    trainer = pl.Trainer(
        gpus=None,
        max_steps=50,
        checkpoint_callback=False,
    )
    trainer.fit(pl_module, train_loader, valid_loader)
