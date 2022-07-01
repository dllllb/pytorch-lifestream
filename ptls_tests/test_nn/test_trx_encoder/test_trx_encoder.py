import pickle

import torch

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder import TrxEncoder


def test_example():
    B, T = 5, 20
    trx_encoder = TrxEncoder(
        embeddings={'mcc_code': {'in': 100, 'out': 5}},
        numeric_values={'amount': 'log'},
    )
    x = PaddedBatch(
        payload={
            'mcc_code': torch.randint(0, 99, (B, T)),
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, 6)  # B, T, H


def test_pickle():
    params = {
        "norm_embeddings": False,
        "embeddings_noise": .1,
        'embeddings': {
            'mcc_code': {'in': 21, 'out': 2},
            'trans_type': {'in': 11, 'out': 2},
        },
        'numeric_values': {'amount': 'log'}
    }

    te = TrxEncoder(**params)
    out = pickle.dumps(te)
    assert len(out) > 0
