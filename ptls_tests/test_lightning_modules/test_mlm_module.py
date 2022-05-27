import torch
from omegaconf import OmegaConf
from ptls.lightning_modules.mlm_module import MLMPretrainModule
from ptls.trx_encoder import PaddedBatch


def get_config():
    return OmegaConf.create(dict(
        hidden_size=8,
        loss_temperature=1.0,
        # norm_predict=False,
        # num_attention_heads=1,
        # intermediate_size=32,
        # num_hidden_layers=1,
        # attention_window=4,
        # max_position_embeddings=100,
        # replace_proba=0.1,
        neg_count=2,
        # max_lr=0.001,
        # weight_decay=0,
        total_steps=100,
        # pct_start=0.2,
    ))


def test_neg_ix():
    m = MLMPretrainModule(
        trx_encoder=None,
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
    m = MLMPretrainModule(
        trx_encoder=lambda x: x,
        **get_config(),
    )

    x = PaddedBatch(
        torch.randn(16, 9, 8),
        torch.randint(3, 9, (16,)),
    )

    loss = m.training_step(x, 0)
    assert loss.item() is not None
