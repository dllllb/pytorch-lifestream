import torch

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_encoder import LongformerEncoder


def test_transformer_encoder_example():
    model = LongformerEncoder(input_size=32)
    x = PaddedBatch(torch.randn(10, 128, 32), torch.randint(20, 128, (10,)))
    y = model(x)
    assert y.payload.size() == (10, 128, 32)

    model = LongformerEncoder(input_size=32, is_reduce_sequence=True)
    y = model(x)
    assert y.size() == (10, 32)


def test_transformer_embedding_size():
    model = LongformerEncoder(input_size=32)
    assert model.embedding_size == 32


def test_transformer_params():
    x = PaddedBatch(torch.randn(10, 128, 32), torch.randint(20, 128, (10,)))
    models = [
        LongformerEncoder(input_size=32, use_positional_encoding=False),
        LongformerEncoder(input_size=32, use_start_random_shift=False),
    ]
    for model in models:
        y = model(x)
        assert y.payload.size() == (10, 128, 32)
