import pytorch_lightning as pl
from ptls.seq_to_target import SequenceToTarget
from .test_data_load import RandomEventData
from pyhocon import ConfigFactory
from ptls.seq_to_target import LogAccuracy
import torch


def tst_params_rnn():
    params = {
        "data_module": {
            "train": {
                "batch_size": 32,
                "num_workers": 1,
                "trx_dropout": .1,
                "max_seq_len": 30
            },
            "valid": {
                "batch_size": 32,
                "num_workers": 1,
                "max_seq_len": 30
            },

        },
        "encoder_type": "rnn",
        'score_metric': 'auroc',
        "train": {
            "weight_decay": 0,
            "lr": 0.004,
            "loss": "bce",
            "random_neg": True,
            "batch_size": 32,
            "num_workers": 1,
            "trx_dropout": .1,
            "n_epoch": 1,
            "max_seq_len": 30
        },
        "valid": {
            "batch_size": 32,
            "num_workers": 1,
            "max_seq_len": 30
        },
        'rnn': {
            'trainable_starter': 'static',
            'hidden_size': 16,
            'type': 'gru',
            'bidir': False,
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
        'head_layers': [
            ["Linear", {"in_features": "{seq_encoder.embedding_size}", "out_features": 1}],
            ["Sigmoid", {}],
            ["Squeeze", {}],
        ],
        "lr_scheduler": {
            "step_size": 10,
            "step_gamma": 0.8
        }
    }
    return ConfigFactory.from_dict(params)


def tst_params_transf():
    params = {
        'model_type': 'transf',
        'transf': {
            'n_heads': 2,
            'dim_hidden': 16,
            'dropout': .1,
            'n_layers': 2,
            'max_seq_len': 200,
            'use_after_mask': True,
            'use_positional_encoding': True,
            'sum_output': True,
            'input_size': 16,
            'train_starter': True,
            'shared_layers': False,
            'use_src_key_padding_mask': False
        },
        "head": {
            "explicit_lengths": False,
        }
    }

    return ConfigFactory.from_dict(params).with_fallback(tst_params_rnn())


def test_train_loop_rnn():
    params = tst_params_rnn()

    model = SequenceToTarget(params)
    dl = RandomEventData(params['data_module'])
    trainer = pl.Trainer(max_epochs=1, logger=None, checkpoint_callback=False)
    trainer.fit(model, dl)


def test_train_loop_transf():
    params = tst_params_transf()

    model = SequenceToTarget(params)
    dl = RandomEventData(params['data_module'])
    trainer = pl.Trainer(max_epochs=1, logger=None, checkpoint_callback=False)
    trainer.fit(model, dl)


def test_accuracy_bin():
    acc = LogAccuracy()
    y_hat = torch.tensor([0.1, 0.4, 0.6, 0.8, 0.9])
    y = torch.tensor([0, 1, 0, 1, 0])
    acc(y_hat, y)
    assert acc.compute() == 0.4


def test_accuracy_mul():
    acc = LogAccuracy()
    y_hat = torch.log_softmax(torch.tensor([
        [-1, 2, 1],
        [1, 2, -1],
        [1, -2, 0],
        [1, 1, 2],
        [1, 1, 2],
    ]).float(), dim=1)
    y = torch.tensor([0, 1, 1, 2, 2])
    acc(y_hat, y)
    assert acc.compute() == 0.6
