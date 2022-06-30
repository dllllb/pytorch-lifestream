import torch
from omegaconf import OmegaConf

from ptls.nn.trx_encoder import TrxEncoder
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_encoder import RnnSeqEncoder, TransformerSeqEncoder
from hydra.utils import instantiate


def get_data():
    payload = {
        'amount': torch.arange(4*10).view(4, 10).float(),
        'event_time': torch.arange(4*10).view(4, 10).float(),
        'mcc_code': torch.arange(4*10).view(4, 10),
        'tr_type': torch.arange(4*10).view(4, 10)
    }
    return PaddedBatch(
        payload=payload,
        length=torch.tensor([4, 2, 6, 8])
    )


def test_rnn_shape():
    model = RnnSeqEncoder(
        trx_encoder=TrxEncoder(
            embeddings={
                'mcc_code': {'in': 200, 'out': 48},
                'tr_type': {'in': 100, 'out': 24},
            },
            numeric_values={'amount': 'identity'},
        ),
        hidden_size=48,
    )

    x = get_data()

    out = model(x)
    assert isinstance(out, torch.Tensor) and out.shape == torch.Size([4, 48])


def test_rnn_hydra_init():
    params = """
seq_encoder: 
    _target_: ptls.nn.RnnSeqEncoder
    trx_encoder:
        _target_: ptls.nn.TrxEncoder
        embeddings:
            mcc_code:
                in: 200
                out: 48
            tr_type:
                in: 100
                out: 24
        numeric_values:
            amount: identity
    hidden_size: 48
    """
    model = instantiate(OmegaConf.create(params))['seq_encoder']

    x = get_data()
    out = model(x)

    assert isinstance(out, torch.Tensor) and out.shape == torch.Size([4, 48])


def test_transformer_shape():
    model = TransformerSeqEncoder(
        trx_encoder=TrxEncoder(
            embeddings={
                'mcc_code': {'in': 200, 'out': 48},
                'tr_type': {'in': 100, 'out': 23},
            },
            numeric_values={'amount': 'identity'},
        ),
        n_heads=2,
        dim_hidden=128,
        n_layers=2,
        max_seq_len=400,
    )

    x = get_data()

    out = model(x)
    assert isinstance(out, torch.Tensor) and out.shape == torch.Size([4, 72])


def test_transformer_hydra_init():
    params = """
seq_encoder: 
    _target_: ptls.nn.TransformerSeqEncoder
    trx_encoder:
        _target_: ptls.nn.TrxEncoder
        embeddings:
            mcc_code:
                in: 200
                out: 48
            tr_type:
                in: 100
                out: 23
        numeric_values:
            amount: identity
    n_heads: 2
    dim_hidden: 128
    n_layers: 2
    max_seq_len: 400
    """
    model = instantiate(OmegaConf.create(params))['seq_encoder']

    x = get_data()
    out = model(x)

    assert isinstance(out, torch.Tensor) and out.shape == torch.Size([4, 72])


def test_transformer_hydra_init_with_sequential():
    params = """
seq_encoder: 
    _target_: ptls.nn.TransformerSeqEncoder
    trx_encoder:
        _target_: torch.nn.Sequential
        _args_:
        -
            _target_: ptls.nn.TrxEncoder
            embeddings:
                mcc_code:
                    in: 200
                    out: 48
                tr_type:
                    in: 100
                    out: 24
            numeric_values:
                amount: identity
        -
            _target_: ptls.nn.PBLinear
            in_features: 73
            out_features: 24
    input_size: 24
    n_heads: 2
    dim_hidden: 128
    n_layers: 2
    max_seq_len: 400
    """
    model = instantiate(OmegaConf.create(params))['seq_encoder']

    x = get_data()
    out = model(x)

    assert isinstance(out, torch.Tensor) and out.shape == torch.Size([4, 24])
