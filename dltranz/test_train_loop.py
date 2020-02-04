import torch
from pyhocon import ConfigFactory
from sklearn.metrics import roc_auc_score

from dltranz.data_load import create_validation_loader, TrxDataset
from dltranz.experiment import fit_model_on_data, run_experiment
from dltranz.models import rnn_model, skip_rnn2_model, transformer_model
from dltranz.test_trx_encoder import gen_trx_data
from dltranz.train import score_model


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


def test_train_loop_rnn():
    params = tst_params()

    train_data = gen_trx_data((torch.rand(1000)*60+1).long())
    train_ds = TrxDataset(train_data)
    test_data = gen_trx_data((torch.rand(100)*60+1).long())
    valid_ds = TrxDataset(test_data)

    res = run_experiment(train_ds, valid_ds, {'params': params}, rnn_model)
    print(res)


def test_score_model():
    params = tst_params()

    test_data = gen_trx_data((torch.rand(1000)*60+1).long())
    valid_loader = create_validation_loader(TrxDataset(test_data), params['valid'])

    pred, true = score_model(rnn_model(params), valid_loader, params)
    print(roc_auc_score(pred, true))


def test_train_loop_skip_rnn():
    params = tst_params_skip_rnn()

    train_data = gen_trx_data((torch.rand(1000) * 60 + 1).long())
    train_ds = TrxDataset(train_data)
    test_data = gen_trx_data((torch.rand(100) * 60 + 1).long())
    valid_ds = TrxDataset(test_data)

    res = run_experiment(train_ds, valid_ds, {'params': params}, rnn_model)
    print(res)


def test_train_loop_transf():
    params = tst_params_transf()

    train_data = gen_trx_data((torch.rand(1000) * 60 + 1).long())
    train_ds = TrxDataset(train_data)
    test_data = gen_trx_data((torch.rand(100) * 60 + 1).long())
    valid_ds = TrxDataset(test_data)

    res = run_experiment(train_ds, valid_ds, {'params': params}, rnn_model)
    print(res)
