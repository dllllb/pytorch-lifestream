import torch
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn import (
    PBLinear, PBL2Norm, PBLayerNorm, PBReLU,
)


def test_pb_doc_example():
    x = PaddedBatch(torch.randn(4, 12, 8), torch.LongTensor([3, 12, 8]))
    model = torch.nn.Sequential(
        PBLinear(8, 5),
        PBReLU(),
        PBLinear(5, 10),
    )
    y = model(x)
    assert y.payload.size() == (4, 12, 10)


def test_pb_linear_args():
    x = PaddedBatch(torch.randn(4, 12, 8), torch.LongTensor([3, 12, 8]))
    model = PBLinear(8, 5)
    y = model(x)
    assert y.payload.size() == (4, 12, 5)
    help(PBLinear)


def test_pb_linear_kwargs():
    x = PaddedBatch(torch.randn(4, 12, 8), torch.LongTensor([3, 12, 8]))
    model = PBLinear(in_features=8, out_features=5)
    y = model(x)
    assert y.payload.size() == (4, 12, 5)


def test_pb_layer_norm():
    x = PaddedBatch(torch.randn(4, 12, 8), torch.LongTensor([3, 12, 8]))
    model = PBLayerNorm(8)
    y = model(x)
    assert y.payload.size() == (4, 12, 8)


def test_pb_l2_norm():
    x = PaddedBatch(torch.randn(4, 12, 8), torch.LongTensor([3, 12, 8]))
    model = PBL2Norm()
    y = model(x)
    assert y.payload.size() == (4, 12, 8)
    torch.testing.assert_allclose(torch.ones(4, 12), y.payload.pow(2).sum(dim=2))


def test_pb_relu():
    x = PaddedBatch(torch.randn(4, 12, 8), torch.LongTensor([3, 12, 8]))
    model = PBReLU()
    y = model(x)
    assert y.payload.size() == (4, 12, 8)
