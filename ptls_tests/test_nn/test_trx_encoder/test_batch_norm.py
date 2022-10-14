import torch

from ptls.data_load import PaddedBatch
from ptls.nn.trx_encoder.batch_norm import RBatchNorm, RBatchNormWithLens


def test_r_batch_norm():
    m = RBatchNorm(1)
    x = PaddedBatch(
        torch.randn(20, 400, 1),
        torch.arange(20, dtype=torch.long) + 10,
    )
    out = m(x).payload
    assert out.size() == (20, 400, 1)


def test_r_batch_norm_with_lens():
    m = RBatchNormWithLens(1)
    x = PaddedBatch(
        torch.randn(20, 400, 1),
        torch.arange(20, dtype=torch.long) + 10,
    )
    out = m(x).payload
    assert out.size() == (20, 400, 1)
