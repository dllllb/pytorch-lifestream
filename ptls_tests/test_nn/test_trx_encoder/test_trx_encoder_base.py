import pytest
import torch

from ptls.data_load import PaddedBatch
from ptls.nn.trx_encoder.scalers import IdentityScaler
from ptls.nn.trx_encoder.trx_encoder_base import TrxEncoderBase


def test_get_category_indexes_clip():
    m = TrxEncoderBase(
        embeddings={'mcc': {'in': 4, 'out': 5}},
    )
    x = PaddedBatch({
            'mcc': torch.arange(6).unsqueeze(0),
        },
        torch.tensor([6]),
    )
    out = m.get_category_indexes(x, 'mcc')
    exp = torch.LongTensor([[0, 1, 2, 3, 3, 3]])
    torch.testing.assert_close(out, exp)


def test_get_category_indexes_assert():
    m = TrxEncoderBase(
        embeddings={'mcc': {'in': 4, 'out': 5}},
        out_of_index='assert',
    )
    x = PaddedBatch({
            'mcc': torch.arange(6).unsqueeze(0),
        },
        torch.tensor([6]),
    )
    with pytest.raises(IndexError):
        _ = m.get_category_indexes(x, 'mcc')


def test_get_category_embeddings():
    m = TrxEncoderBase(
        embeddings={'mcc': {'in': 4, 'out': 5}},
    )
    x = PaddedBatch({
            'mcc': torch.arange(6).unsqueeze(0),
        },
        torch.tensor([6]),
    )
    out = m.get_category_embeddings(x, 'mcc')
    assert out.size() == (1, 6, 5)
    torch.testing.assert_close(out[0, 3], out[0, 5])


def test_numeric_scaler():
    m = TrxEncoderBase(
        numeric_values={'amount': 'log'},
    )
    x = PaddedBatch({
            'amount': torch.arange(6).unsqueeze(0),
        },
        torch.tensor([6]),
    )
    out = m.get_custom_embeddings(x, 'amount')
    torch.testing.assert_close(out.expm1(), torch.arange(6).float().view(1, 6, 1))


def test_numeric_scaler_with_alias():
    m = TrxEncoderBase(
        numeric_values={'amount_x': IdentityScaler(col_name='amount')},
    )
    x = PaddedBatch({
            'amount': torch.arange(6).unsqueeze(0),
        },
        torch.tensor([6]),
    )
    out = m.get_custom_embeddings(x, 'amount_x')
    torch.testing.assert_close(out, torch.arange(6).float().view(1, 6, 1))


def test_output_size():
    m = TrxEncoderBase(
        embeddings={'mcc': {'in': 4, 'out': 5}},
        numeric_values={'amount': 'log'},
    )
    assert m.numerical_size == 1
    assert m.embedding_size == 5
    assert m.output_size == 6


def test_disable_embedding1():
    m = TrxEncoderBase(
        embeddings={
            'mcc': {'in': 4, 'out': 5},
            'code': {'in': 0}
        },
    )
    assert m.embedding_size == 5


def test_disable_embedding2():
    m = TrxEncoderBase(
        embeddings={
            'mcc': {'in': 4, 'out': 5},
            'code': {'in': 8, 'out': 0},
        },
    )
    assert m.embedding_size == 5


def test_disable_embedding3():
    m = TrxEncoderBase(
        embeddings={
            'mcc': {'in': 4, 'out': 5},
            'code': {'disabled': True},
        },
    )
    assert m.embedding_size == 5


def test_disable_numerical():
    m = TrxEncoderBase(
        numeric_values={
            'amount': 'log',
            'amount_log': 'none',
        },
    )
    assert m.numerical_size == 1


def test_output_size_extended():
    m = TrxEncoderBase(
        embeddings={
            'mcc': {'in': 4, 'out': 5},
            'currency': torch.nn.Embedding(5, 3),
        },
        numeric_values={
            'amount': 'log',
            'amount_log': IdentityScaler(col_name='amount'),
        },
    )
    assert m.numerical_size == 2
    assert m.embedding_size == 8
    assert m.output_size == 10

def test_category_max_size():
    m = TrxEncoderBase(
        embeddings={
            'mcc': {'in': 4, 'out': 5},
            'currency': torch.nn.Embedding(5, 3),
        },
        numeric_values={
            'amount': 'log',
            'amount_log': IdentityScaler(col_name='amount'),
        },
    )
    category_max_size = m.category_max_size
    assert category_max_size['mcc'] == 4
    assert category_max_size['currency'] == 5

