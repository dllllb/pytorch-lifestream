import torch
from omegaconf import OmegaConf
from ptls.frames.bert.modules.mlm_module import MLMPretrainModule
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn import RnnEncoder, TransformerEncoder, LongformerEncoder


def get_config():
    return OmegaConf.create(dict(
        hidden_size=8,
        loss_temperature=1.0,
        neg_count=2,
        total_steps=100,
    ))


def test_neg_ix():
    m = MLMPretrainModule(
        trx_encoder=None,
        seq_encoder=AbsSeqEncoder(),
        **OmegaConf.merge(get_config(), OmegaConf.from_dotlist(["neg_count=4"])),
    )
    mask = torch.Tensor([
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]).bool()

    b_ix, neg_ix = m.get_neg_ix(mask)
    assert b_ix.size() == (5, 4)
    assert neg_ix.size() == (5, 4)
    assert mask[b_ix, neg_ix].all()


def test_training_step():
    models = [
        RnnEncoder(input_size=8, hidden_size=8),
        TransformerEncoder(input_size=8),
        LongformerEncoder(input_size=8),

    ]
    for seq_encoder in models:
        m = MLMPretrainModule(
            trx_encoder=lambda x: x,
            seq_encoder=seq_encoder,
            **get_config(),
        )

        x = PaddedBatch(
            torch.randn(16, 9, 8),
            torch.randint(3, 9, (16,)),
        )

        loss = m.training_step(x, 0)
        assert loss.item() is not None
