import pickle

import torch

from dltranz.data_load import padded_collate, TrxDataset
from dltranz.trx_encoder import TrxEncoder, TrxMeanEncoder
from .test_data_load import gen_trx_data


def test_simple():
    x, y = padded_collate(TrxDataset(gen_trx_data([4, 3, 2])))

    params = {
        "norm_embeddings": False,
        "embeddings_noise": .1,
        'embeddings': {
            'mcc_code': {'in': 21, 'out': 2},
            'trans_type': {'in': 11, 'out': 2},
        },
        'numeric_values': {'amount': 'log'}
    }

    te = TrxEncoder(params)

    e = te(x)

    assert e.payload.shape == (3, 4, 5)


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

    te = TrxEncoder(params)
    out = pickle.dumps(te)
    assert len(out) > 0


def test_mean_encoder():
    x, y = padded_collate(TrxDataset(gen_trx_data([4, 3, 2])))

    params = {
        "embeddings_noise": .1,
        'embeddings': {
            'mcc_code': {'in': 21, 'out': 2},
            'trans_type': {'in': 11, 'out': 2},
        },
        'numeric_values': {'amount': 'log'}
    }

    te = TrxMeanEncoder(params)

    e = te(x)

    assert e.shape == (3, 33)
