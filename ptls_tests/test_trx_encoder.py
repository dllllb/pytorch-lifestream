import pickle

import torch

from ptls.data_load import padded_collate, TrxDataset
from ptls.trx_encoder import PaddedBatch, TrxEncoder, TrxMeanEncoder
from .test_data_load import gen_trx_data


def test_padded_batch_mask_tensor_trx_embedding():
    data = PaddedBatch(torch.randn(4, 5, 3), torch.tensor([2, 5, 1, 3]))
    out = data.seq_len_mask
    exp = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
    ]).long()
    torch.testing.assert_close(out, exp)


def test_padded_batch_mask_tensor_numerical():
    data = PaddedBatch(torch.randn(4, 5), torch.tensor([2, 5, 3, 1]))
    out = data.seq_len_mask
    exp = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
    ]).long()
    torch.testing.assert_close(out, exp)


def test_padded_batch_mask_dict():
    data = PaddedBatch({'col1': torch.randn(4, 5), 'col2': torch.randn(4, 5)}, torch.tensor([4, 5, 1, 3]))
    out = data.seq_len_mask
    exp = torch.tensor([
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0]
    ]).long()
    torch.testing.assert_close(out, exp)


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
