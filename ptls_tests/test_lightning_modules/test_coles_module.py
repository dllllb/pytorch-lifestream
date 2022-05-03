import pytorch_lightning as pl
from ptls.lightning_modules.coles_module import CoLESModule
from ..test_data_load import RandomEventData
from pyhocon import ConfigFactory


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
        "validation_metric_params": {
            "K": 2,
            "metric": "cosine"
        },
        "encoder_type": "rnn",
        "train": {
            "sampling_strategy": "HardNegativePair",
            "neg_count": 5,
            "loss": "MarginLoss",
            "margin": 0.2,
            "beta": 0.4,
            "lr": 0.002,
            "weight_decay": 0.0,
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
        "head_layers": [
            ["NormEncoder", {}],
        ],
        "lr_scheduler": {
            "step_size": 10,
            "step_gamma": 0.9025
        },
    }

    params = ConfigFactory.from_dict(params)
    return params


def test_train_loop():
    params = tst_params()

    model = CoLESModule(params)
    dl = RandomEventData(params['data_module'])
    trainer = pl.Trainer(max_epochs=1, logger=None, checkpoint_callback=False)
    trainer.fit(model, dl)
