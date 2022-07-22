import torch

from ptls.data_load import PaddedBatch
from ptls.nn.trx_encoder.batch_norm import RBatchNorm, RBatchNormWithLens


def test_r_batch_norm():
    m = RBatchNorm()
    x = torch.randn(20, 400)
    out = m(x)
    assert out.size() == (20, 400, 1)


def test_r_batch_norm_with_lens():
    m = RBatchNormWithLens()
    x = PaddedBatch(
        torch.randn(20, 400),
        torch.arange(20, dtype=torch.long) + 10,
    )
    out = m(x)
    assert out.size() == (20, 400, 1)
