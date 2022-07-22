from functools import partial

import omegaconf
import pytorch_lightning as pl
import torch
import hydra
import torchmetrics
from pyhocon import ConfigFactory

from ptls.frames.supervised import SequenceToTarget
from ptls.frames.supervised.metrics import LogAccuracy, UnivMeanError, BucketAccuracy, JSDiv
from ptls.loss import BCELoss, ZILNLoss
from ptls.nn import PBLinear, RnnSeqEncoder, TransformerSeqEncoder, TrxEncoder
from ptls_tests.test_data_load import RandomEventData


def get_rnn_params():
    return dict(
        seq_encoder=RnnSeqEncoder(
            trx_encoder=TrxEncoder(
                norm_embeddings=False,
                embeddings_noise=0.1,
                embeddings={
                    'mcc_code': {'in': 21, 'out': 3},
                    'trans_type': {'in': 11, 'out': 2},
                },
                numeric_values={'amount': 'log'},
            ),
            hidden_size=16,
            type='gru',
        ),
        optimizer_partial=partial(torch.optim.Adam, lr=0.004),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.8),
    )


def tst_params_data():
    params = {
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

    return ConfigFactory.from_dict(params).with_fallback(tst_params_data())


def test_train_loop_rnn_binary_classification():
    model = SequenceToTarget(
        head=torch.nn.Sequential(
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),
            torch.nn.Flatten(start_dim=0),
        ),
        loss=BCELoss(),
        metric_list=torchmetrics.AUROC(num_classes=2),
        **get_rnn_params(),
    )
    dl = RandomEventData(tst_params_data(), target_type='bin_cls')
    trainer = pl.Trainer(max_epochs=1, logger=None, checkpoint_callback=False)
    trainer.fit(model, dl)
    print(trainer.logged_metrics)


def test_train_loop_rnn_milti_classification():
    model = SequenceToTarget(
        head=torch.nn.Sequential(
            torch.nn.Linear(16, 4),
            torch.nn.LogSoftmax(dim=1),
        ),
        loss=torch.nn.NLLLoss(),
        metric_list={
            'auroc': torchmetrics.AUROC(num_classes=4),
            'accuracy': torchmetrics.Accuracy(),
        },
        **get_rnn_params(),
    )
    dl = RandomEventData(tst_params_data(), target_type='multi_cls')
    trainer = pl.Trainer(max_epochs=1, logger=None, checkpoint_callback=False)
    trainer.fit(model, dl)
    print(trainer.logged_metrics)


def test_train_loop_rnn_regression():
    model = SequenceToTarget(
        head=torch.nn.Sequential(
            torch.nn.Linear(16, 1),
            torch.nn.Flatten(start_dim=0),
        ),
        loss=torch.nn.MSELoss(),
        metric_list=torchmetrics.MeanSquaredError(compute_on_step=False, squared=False),
        **get_rnn_params(),
    )
    dl = RandomEventData(tst_params_data(), target_type='regression')
    trainer = pl.Trainer(max_epochs=1, logger=None, checkpoint_callback=False)
    trainer.fit(model, dl)
    print(trainer.logged_metrics)


def test_train_loop_transf():
    trx_encoder = TrxEncoder(
        norm_embeddings=False,
        embeddings_noise=0.1,
        embeddings={
            'mcc_code': {'in': 21, 'out': 3},
            'trans_type': {'in': 11, 'out': 2},
        },
        numeric_values={'amount': 'log'},
    )
    seq_encoder = TransformerSeqEncoder(
        trx_encoder=torch.nn.Sequential(trx_encoder, PBLinear(trx_encoder.output_size, 20)),
        input_size=20,
        n_heads=1,
        n_layers=1,
    )
    model = SequenceToTarget(
        seq_encoder=seq_encoder,
        head=torch.nn.Sequential(
            torch.nn.Linear(seq_encoder.embedding_size, 1),
            torch.nn.Sigmoid(),
            torch.nn.Flatten(start_dim=0),
        ),
        loss=BCELoss(),
        metric_list=torchmetrics.AUROC(num_classes=2),
        optimizer_partial=partial(torch.optim.Adam, lr=0.004),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.8),
    )
    dl = RandomEventData(tst_params_data())
    trainer = pl.Trainer(max_epochs=1, logger=None, checkpoint_callback=False)
    trainer.fit(model, dl)

# SequenceToTarget.metric_list
def test_seq_to_target_metric_list_single_metric():
    model = SequenceToTarget(metric_list=torchmetrics.Accuracy(), seq_encoder=None)
    metric_name = next(iter(model.valid_metrics.keys()))
    assert metric_name == 'Accuracy'


def test_seq_to_target_metric_list_list_with_metric():
    model = SequenceToTarget(metric_list=[
        torchmetrics.Accuracy(),
        torchmetrics.AUROC(num_classes=2),
    ], seq_encoder=None)
    assert 'Accuracy' in model.valid_metrics
    assert 'AUROC' in model.valid_metrics


def test_seq_to_target_metric_list_dict_with_single_metric():
    model = SequenceToTarget(metric_list={
        'acc': torchmetrics.Accuracy(),
    }, seq_encoder=None)
    metric_name = next(iter(model.valid_metrics.keys()))
    assert metric_name == 'acc'


def test_seq_to_target_metric_list_dict_with_metric():
    model = SequenceToTarget(metric_list={
        'acc': torchmetrics.Accuracy(),
        'auroc': torchmetrics.AUROC(num_classes=2),
    }, seq_encoder=None)
    assert 'acc' in model.valid_metrics
    assert 'auroc' in model.valid_metrics


def test_seq_to_target_metric_list_dict_config_with_metric():
    conf = omegaconf.OmegaConf.create("""
        auroc:
            _target_: torchmetrics.AUROC
            num_classes: 2
        acc:
            _target_: torchmetrics.Accuracy
    """)
    model = SequenceToTarget(metric_list=hydra.utils.instantiate(conf), seq_encoder=None)
    assert 'acc' in model.valid_metrics
    assert 'auroc' in model.valid_metrics


def test_accuracy_bin():
    acc = LogAccuracy()
    y_hat = torch.tensor([0.1, 0.4, 0.6, 0.8, 0.9])
    y = torch.tensor([0, 1, 0, 1, 0])
    acc(y_hat, y)
    assert acc.compute().mul(100).int() == 40


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
    assert acc.compute().mul(100).int() == 60


def test_ziln_loss():
    ziln = ZILNLoss()
    min_loss = 0.5 * torch.log(torch.tensor(ziln.eps))
    y = torch.zeros(10, 5)
    assert ziln(torch.zeros(y.shape[0]), y) == 0
    assert ziln(torch.zeros(y.shape[0], 2), y) >= min_loss
    assert ziln(torch.zeros(y.shape[0], 3), y) >= min_loss
    assert ziln(torch.zeros(y.shape[0], 3 + y.shape[1]), y) >= min_loss


def test_bucket_accuracy():
    bacc, B = BucketAccuracy(), 100
    assert bacc(torch.randn(B), torch.randn(B)) >= 0
    assert bacc(torch.randn(B, 3), torch.randn(B)) >= 0
    assert bacc(torch.randn(B), torch.randn(B, 3)) >= 0
    assert bacc(torch.randn(B, 3), torch.randn(B, 3)) >= 0


def test_rmse():
    rmse, B = UnivMeanError(), 10
    assert rmse(torch.randn(B), torch.randn(B)) >= 0
    assert rmse(torch.randn(B, 3), torch.randn(B)) >= 0
    assert rmse(torch.randn(B, 3), torch.randn(B, 3)) >= 0


def test_jsdiv():
    div = JSDiv()
    y = torch.zeros(10, 5)
    assert div(torch.randn(y.shape[0], y.shape[1] + 3), y) >= 0
