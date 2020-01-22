import torch
from dltranz.seq_encoder import RnnEncoder
from dltranz.trx_encoder import TrxEncoder
from dltranz.cpc import CPC_Ecoder, CPC_Loss, run_experiment
from dltranz.test_trx_encoder import gen_trx_data
from dltranz.data_load import TrxDataset


def tst_rnn_model(config):
    config = {
        'rnn': {
            'trainable_starter': 'empty',
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
        "train": {
            "weight_decay": 0,
            "lr": 0.004,
            "lr_scheduler.step_size": 10,
            "lr_scheduler.step_gamma": 0.8,
            "batch_size": 32,
            "num_workers": 1,
            "trx_dropout": .1,
            "n_epoch": 1,
            "max_seq_len": 30,
            "cpc": {
                "n_forward_steps": 3,
                "linear_size": 16
            }
        },
        "valid": {
            "batch_size": 32,
            "num_workers": 1,
            "max_seq_len": 30
        },
    }

    trx_e = TrxEncoder(config['trx_encoder'])
    rnn_e = RnnEncoder(8, config['rnn'])
    cpc_e = CPC_Ecoder(trx_e, rnn_e)

    train_data = gen_trx_data((torch.rand(1000)*60+1).long())
    train_ds = TrxDataset(train_data)
    test_data = gen_trx_data((torch.rand(100)*60+1).long())
    valid_ds = TrxDataset(test_data)

    run_experiment(train_ds, valid_ds, cpc_e, config)
