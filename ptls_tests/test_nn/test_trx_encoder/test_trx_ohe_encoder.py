import pickle

import torch

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder import TrxEncoderOhe


def test_example():
    B, T = 5, 20
    trx_encoder = TrxEncoderOhe(
        embeddings={'mcc_code': {'in': 100}},
        numeric_values={'amount': 'log'},
    )
    x = PaddedBatch(
        payload={
            'mcc_code': torch.randint(1, 99, (B, T)),
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, 100)  # B, T, H
    assert trx_encoder.output_size == 100
    z_sum = z.payload[:, :, :-1].sum(dim=2)
    assert (z_sum == 1).all()
