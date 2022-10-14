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
    assert trx_encoder.output_size == 6


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


def test_linear_projection():
    te = TrxEncoder(
        embeddings={
            'mcc_code': {'in': 21, 'out': 2},
            'trans_type': {'in': 11, 'out': 2},
        },
        numeric_values={'amount': 'log'},
        linear_projection_size=11,
    )
    out = te(PaddedBatch({
        'mcc_code': torch.randint(0, 10, (4, 8)),
        'trans_type': torch.randint(0, 10, (4, 8)),
        'amount': torch.randn(4, 8),
    }, torch.LongTensor([10, 4, 9, 2])))
    assert out.payload.size() == (4, 8, 11)
    assert te.output_size == 11


def test_orthogonal_init():
    te = TrxEncoder(
        embeddings={
            'mcc_code': {'in': 5, 'out': 4},
        },
        orthogonal_init=True,
    )
    out = te(PaddedBatch({
        'mcc_code': torch.arange(5).unsqueeze(0),
    }, torch.LongTensor([5]))).payload[0]

    torch.testing.assert_close(
        torch.einsum('ij,kj->ik', out[1:], out[1:]),
        torch.eye(4),
    )


def test_orthogonal_init_with_linear_head():
    te = TrxEncoder(
        embeddings={
            'mcc_code': {'in': 5, 'out': 4},
        },
        linear_projection_size=4,
        orthogonal_init=True,
    )
    out = te.linear_projection_head.weight.data
    torch.testing.assert_close(
        torch.einsum('ij,kj->ik', out, out),
        torch.eye(4),
    )


def test_batch_norm_none():
    te = TrxEncoder(
        numeric_values={
            'amount': 'identity',
        },
        use_batch_norm=False,
    )
    x = PaddedBatch({
        'amount': torch.FloatTensor([
            [1, 2, -1, 3, 0, 0],
            [-2, 1, 0, 0, 0, 0],
        ]),
    }, torch.LongTensor([4, 2]))
    out = te(x).payload

    assert out[0, 0] == 1


def test_batch_norm_batch():
    te = TrxEncoder(
        numeric_values={
            'amount': 'identity',
        },
        use_batch_norm=True,
        use_batch_norm_with_lens=False,
    )
    x = PaddedBatch({
        'amount': torch.FloatTensor([
            [1, 2, -1, 3, 0, 0],
            [-2, 1, 0, 0, 0, 0],
        ]),
    }, torch.LongTensor([4, 2]))
    out = te(x).payload

    flat = torch.FloatTensor([1, 2, -1, 3, 0, 0, -2, 1, 0, 0, 0, 0])
    mean = flat.mean()
    std = flat.std()
    assert abs(out[0, 0] - (1 - mean) / std) < 0.1


def test_batch_norm_with_lens():
    te = TrxEncoder(
        numeric_values={
            'amount': 'identity',
        },
        use_batch_norm=True,
        use_batch_norm_with_lens=True,
    )
    x = PaddedBatch({
        'amount': torch.FloatTensor([
            [1, 2, -1, 3, 0, 0],
            [-2, 1, 0, 0, 0, 0],
        ]),
    }, torch.LongTensor([4, 2]))
    out = te(x).payload

    flat = torch.FloatTensor([1, 2, -1, 3, -2, 1])
    mean = flat.mean()
    std = flat.std()
    assert abs(out[0, 0] - (1 - mean) / std) < 0.1


