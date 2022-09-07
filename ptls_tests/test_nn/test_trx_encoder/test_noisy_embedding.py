import torch

from ptls.nn.trx_encoder.noisy_embedding import NoisyEmbedding


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


def test_spatial_dropout():
    embedding = NoisyEmbedding(16, 4, 0, noise_scale=0.0, dropout=0.27, spatial_dropout=True)
    x = torch.ones(4000, 8000, dtype=torch.long)
    out = embedding(x)
    nonzero = torch.count_nonzero(out, dim=1)
    assert torch.all((nonzero == 8000) + (nonzero == 0)).item()
    assert torch.abs(0.27 - (1 - torch.count_nonzero(out) / 4000 / 8000 / 4)) < 0.01


def test_dropout():
    embedding = NoisyEmbedding(16, 4, 0, noise_scale=0.0, dropout=0.35, spatial_dropout=False)
    x = torch.ones(400, 800, dtype=torch.long)
    out = embedding(x)
    assert torch.abs(0.35 - (1 - torch.count_nonzero(out) / 400 / 800 / 4)) < 0.001
