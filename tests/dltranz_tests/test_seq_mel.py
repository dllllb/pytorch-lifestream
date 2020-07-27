import pytorch_lightning as pl
import torch
from dltranz.seq_mel import SequenceMetricLearning
from dltranz.data_load import create_train_loader
from .test_data_load import gen_trx_data

class SequenceMetricLearningTesting(SequenceMetricLearning):
    def train_dataloader(self):
        test_data = gen_trx_data((torch.rand(1000)*60+1).long())
        train_loader = create_train_loader(test_data, self.params)
        return train_loader


def test_train_loop():
    params = {
        "weight_decay": 0,
        "lr": 0.004,
        "batch_size": 32,
        "num_workers": 1,
        "trx_dropout": .1,
        "n_epoch": 1,
        "max_seq_len": 30
    }

    model = SequenceMetricLearning(params)
    trainer = pl.Trainer()
    trainer.fit(model)
