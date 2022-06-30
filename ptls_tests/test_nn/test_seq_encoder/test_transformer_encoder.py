import torch

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_encoder import TransformerEncoder
from ptls.nn.seq_encoder.transformer_encoder import PositionalEncoding


def test_positional_encoding_shape():
    pe = PositionalEncoding(256, max_len=4000)
    assert pe.pe.size() == (1, 4000, 256)


def test_positional_encoding_forward_train():
    pe = PositionalEncoding(256, max_len=4000)
    pe.train()
    x = torch.randn(10, 128, 256)
    y = pe(x)
    assert y.size() == (10, 128, 256)


def test_positional_encoding_forward_eval():
    pe = PositionalEncoding(256, 4000)
    pe.eval()
    x = torch.randn(10, 128, 256)
    y = pe(x)
    assert y.size() == (10, 128, 256)


def test_transformer_encoder_example():
    model = TransformerEncoder(input_size=32)
    x = PaddedBatch(torch.randn(10, 128, 32), torch.randint(20, 128, (10,)))
    y = model(x)
    assert y.payload.size() == (10, 128, 32)

    model = TransformerEncoder(input_size=32, is_reduce_sequence=True)
    y = model(x)
    assert y.size() == (10, 32)


def test_transformer_embedding_size():
    model = TransformerEncoder(input_size=32)
    assert model.embedding_size == 32


def test_transformer_params():
    x = PaddedBatch(torch.randn(10, 128, 32), torch.randint(20, 128, (10,)))
    models = [
        TransformerEncoder(input_size=32, starter='zeros'),
        TransformerEncoder(input_size=32, shared_layers=True),
        TransformerEncoder(input_size=32, use_positional_encoding=False),
        TransformerEncoder(input_size=32, use_start_random_shift=False),
        TransformerEncoder(input_size=32, use_after_mask=True),
        TransformerEncoder(input_size=32, use_src_key_padding_mask=False),
        TransformerEncoder(input_size=32, use_norm_layer=False),
        TransformerEncoder(input_size=32, use_norm_layer=False, shared_layers=True),
    ]
    for model in models:
        y = model(x)
        assert y.payload.size() == (10, 128, 32)
