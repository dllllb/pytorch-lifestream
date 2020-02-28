import torch
from pyhocon import ConfigFactory

from dltranz.agg_feature_model import AggFeatureModel
from dltranz.trx_encoder import PaddedBatch


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
        'params': {
            'trx_encoder': {
                'numeric_values': {
                    'num1': {},
                },
                'embeddings': {
                    'cat1': {'in': 8}
                },
            }
        }
    }
    return ConfigFactory.from_dict(params)


def test_output_size():
    out_size = AggFeatureModel.output_size(get_conf()['params.trx_encoder'])
    assert out_size == 29


def test_model():
    model = AggFeatureModel(get_conf()['params.trx_encoder'])
    out = model(get_data())
    assert out.size() == (3, 29)
