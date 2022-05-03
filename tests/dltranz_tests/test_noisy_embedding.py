import torch

from ptls.trx_encoder import NoisyEmbedding


def test_no_noise():
    embedding = NoisyEmbedding(16, 4, 0, noise_scale=0.0)
    x = torch.zeros(4, 8, dtype=torch.long)
    out = embedding(x)
    assert out.size() == (4, 8, 4)


def test_noisy_embedding():
    embedding = NoisyEmbedding(16, 4, 0, noise_scale=1.0)
    x = torch.zeros(4, 8, dtype=torch.long)
    out = embedding(x)
    assert out.size() == (4, 8, 4)
