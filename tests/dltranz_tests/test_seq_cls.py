import pytorch_lightning as pl
import torch
from dltranz.seq_cls import SequenceClassify
from dltranz.data_load import create_train_loader, create_validation_loader, TrxDataset
from .test_data_load import gen_trx_data
from pyhocon import ConfigFactory


class SequenceClassifyTesting(SequenceClassify):
    def train_dataloader(self):
        test_data = TrxDataset(gen_trx_data((torch.rand(1000)*60+1).long()))
        train_loader = create_train_loader(test_data, self.hparams['train'])
        return train_loader

    def valid_dataloader(self):
        test_data = TrxDataset(gen_trx_data((torch.rand(100)*60+1).long()))
        train_loader = create_validation_loader(test_data, self.hparams['valid'])
        return train_loader

def tst_params_rnn():
    params = {
        "model_type": "rnn",
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

    model = SequenceClassifyTesting(params)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model)

def test_train_loop_transf():
    params = tst_params_transf()

    model = SequenceClassifyTesting(params)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model)