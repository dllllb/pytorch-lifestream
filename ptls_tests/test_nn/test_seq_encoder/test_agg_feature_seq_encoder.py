import torch
from pyhocon import ConfigFactory

from ptls.nn.seq_encoder import AggFeatureSeqEncoder
from ptls.data_load.padded_batch import PaddedBatch


def get_data():
    return PaddedBatch(
        payload={
            'cat1': torch.tensor([
                [1, 3, 5, 0, 0, 0],
                [2, 5, 3, 1, 1, 0],
                [4, 0, 0, 0, 0, 0],
            ]),
            'num1': torch.tensor([
                [5, 6, 3, 0, 0, 0],
                [4, 0, 1, 3, 2, 0],
                [7, 0, 0, 0, 0, 0],
            ]).float(),
        },
        length=torch.tensor([3, 5, 1]),
    )


def get_conf():
    params = {
        'numeric_values': {
            'num1': {},
        },
        'embeddings': {
            'cat1': {'in': 8}
        },
        'was_logified': False,
        'log_scale_factor': 1.0,
        'is_used_count': False,
        'is_used_mean': True,
        'is_used_std': True,
        "use_topk_cnt": 3,
    }
    return ConfigFactory.from_dict(params)


def test_output_size():
    embedding_size = AggFeatureSeqEncoder(**get_conf()).embedding_size
    assert embedding_size == 24


def test_model():
    model = AggFeatureSeqEncoder(**get_conf())
    out = model(get_data())
    assert out.size() == (3, 24)
