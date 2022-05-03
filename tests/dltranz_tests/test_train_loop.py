import numpy as np
import torch
from pyhocon import ConfigFactory
from sklearn.metrics import roc_auc_score

from ptls.data_load import create_validation_loader, TrxDataset
from ptls.models import rnn_model
from .test_data_load import gen_trx_data
from ptls.train import score_model, score_model


def tst_params():
    params = {
        'ensemble_size': 1,
        'score_metric': 'auroc',
        'norm_scores': True,
        'head': {
            "pred_all_states": False,
            "pred_all_states_mean": False,
            "explicit_lengths": True,
            "norm_input": False,
            'use_batch_norm': False,
        },
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
        "lr_scheduler": {
            "step_size": 10,
            "step_gamma": 0.8
        }
    }
    return ConfigFactory.from_dict(params)


def tst_params_skip_rnn():
    params = {
        'skip_rnn': {
            'rnn0': {
                'trainable_starter': 'static',
                'hidden_size': 16,
                'type': 'gru',
                'bidir': False,
            },
            'rnn1': {
                'trainable_starter': 'static',
                'hidden_size': 8,
                'type': 'gru',
                'bidir': False,
            },
            "skip_step_size": 3
        },
        "head": {
            "explicit_lengths": False,
        }
    }

    return ConfigFactory.from_dict(params).with_fallback(tst_params())


def tst_params_transf():
    params = {
        'transf': {
            'n_heads': 2,
            'dim_hidden': 16,
            'dropout': .1,
            'n_layers': 2,
            'max_seq_len': 200,
            'use_after_mask': True,
            'use_positional_encoding': True,
            'sum_output': True,
            'input_size': 16
        },
        "head": {
            "explicit_lengths": False,
        }
    }

    return ConfigFactory.from_dict(params).with_fallback(tst_params())


def tst_params_swa():
    params = {
        'train': {
            'swa': {
                'swa_start': 0
            }
        }
    }

    return ConfigFactory.from_dict(params).with_fallback(tst_params_skip_rnn())


def tst_loaders():
    train_data = gen_trx_data((torch.rand(1000) * 60 + 1).long())
    train_ds = TrxDataset(train_data)
    test_data = gen_trx_data((torch.rand(100) * 60 + 1).long())
    valid_ds = TrxDataset(test_data)
    return train_ds, valid_ds


def test_score_model():
    params = tst_params()

    test_data = gen_trx_data((torch.rand(1000)*60+1).long())
    valid_loader = create_validation_loader(TrxDataset(test_data), params['valid'])

    pred, true = score_model(rnn_model(params), valid_loader, params)
    print(roc_auc_score(true, pred))


def test_score_model_mult1():
    params = tst_params()

    test_data = gen_trx_data((torch.rand(1000)*60+1).long())
    valid_loader = create_validation_loader(TrxDataset(test_data), params['valid'])

    pred, true = score_model(rnn_model(params), valid_loader, params)
    print(roc_auc_score(true, pred))


def test_score_model_mult2():
    model = torch.nn.Sequential(torch.nn.Linear(16, 2), torch.nn.Sigmoid())
    valid_loader = [
        (torch.rand(4, 16), np.arange(4), np.arange(4).astype(str)),
        (torch.rand(2, 16), np.arange(2), np.arange(2).astype(str)),
        (torch.rand(1, 16), np.arange(1), np.arange(1).astype(str)),
    ]

    pred, id1, id2 = score_model(model, valid_loader, {'device': 'cpu'})
    assert pred.shape == (7, 2)
    assert id1.shape == (7,)
    assert id2.shape == (7,)

    np.testing.assert_array_almost_equal(id1, np.array([0, 1, 2, 3, 0, 1, 0]))
    assert id2.tolist() == ['0', '1', '2', '3', '0', '1', '0']
