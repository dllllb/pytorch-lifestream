import torch

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder import TrxEncoder

from ptls.nn.trx_encoder.scalers import Periodic, PeriodicMLP, PLE, PLE_MLP


def test_periodic():
    B, T = 5, 20
    num_periods = 4
    scaler = Periodic(num_periods = num_periods, param_dist_sigma = 3)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, 2 * num_periods)  # B, T, H
    assert trx_encoder.output_size == 2 * num_periods


def test_periodic_mlp():
    B, T = 5, 20
    num_periods = 4
    scaler = PeriodicMLP(num_periods = num_periods, param_dist_sigma = 3)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, 2 * num_periods)  # B, T, H
    assert trx_encoder.output_size == 2 * num_periods


def test_periodic_mlp2():
    B, T = 5, 20
    num_periods = 4
    mlp_output_size = 32
    scaler = PeriodicMLP(num_periods = num_periods, param_dist_sigma = 3, mlp_output_size = mlp_output_size)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, mlp_output_size)  # B, T, H
    assert trx_encoder.output_size == mlp_output_size



def test_ple():
    B, T = 5, 20
    bins = [-1, 0, 1]
    scaler = PLE(bins = bins)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, len(bins) - 1)  # B, T, H
    assert trx_encoder.output_size == len(bins) - 1

def test_ple_mlp():
    B, T = 5, 20
    bins = [-1, 0, 1]
    scaler = PLE_MLP(bins = bins)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, len(bins) - 1)  # B, T, H
    assert trx_encoder.output_size == len(bins) - 1


def test_ple_mlp2():
    B, T = 5, 20
    bins = [-1, 0, 1]
    mlp_output_size = 64
    scaler = PLE_MLP(bins = bins, mlp_output_size = mlp_output_size)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, mlp_output_size)  # B, T, H
    assert trx_encoder.output_size == mlp_output_size

