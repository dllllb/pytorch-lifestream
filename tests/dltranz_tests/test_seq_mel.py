import pytorch_lightning as pl
import torch
from dltranz.seq_mel import SequenceMetricLearning
from dltranz.data_load import create_train_loader, create_validation_loader, TrxDataset
from .test_data_load import gen_trx_data
from pyhocon import ConfigFactory

class SequenceMetricLearningTesting(SequenceMetricLearning):
    def train_dataloader(self):
        test_data = TrxDataset(gen_trx_data((torch.rand(1000)*60+1).long()))
        train_loader = create_train_loader(test_data, self.hparams['train'])
        return train_loader

    def valid_dataloader(self):
        test_data = TrxDataset(gen_trx_data((torch.rand(100)*60+1).long()))
        train_loader = create_validation_loader(test_data, self.hparams['valid'])
        return train_loader

def tst_params():
    params = {
        "model_type": "rnn",
        "train": {
            "num_workers": 1,
            "batch_size": 32,
            "split_strategy": {
                "split_strategy": "SampleSlices",
                "split_count": 3,
                "cnt_min": 3,
                "cnt_max": 75
            },
            "sampling_strategy": "HardNegativePair",
            "trx_dropout": 0.01,
            "max_seq_len": 100,
            "neg_count": 5,
            "loss": "MarginLoss",
            "margin": 0.2,
            "beta": 0.4,
            "lr": 0.002,
            "weight_decay": 0.0,
            "n_epoch": 1,
        },
        "valid": {
            "batch_size": 16,
            "num_workers": 1,
            "split_strategy": {
                "split_strategy": "SampleSlices",
                "split_count": 3,
                "cnt_min": 3,
                "cnt_max": 75
            },
            "max_seq_len": 100
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
        "use_normalization_layer": True,
        "lr_scheduler": {
            "step_size": 10,
            "step_gamma": 0.9025
        },
    }

    params = ConfigFactory.from_dict(params)
    return params

def test_train_loop():
    params = tst_params()

    model = SequenceMetricLearningTesting(params)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model)
