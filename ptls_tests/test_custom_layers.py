import torch
import numpy as np

from ptls.custom_layers import DropoutEncoder, Squeeze, CatLayer, MLP, TabularRowEncoder, CombinedTargetHeadFromRnn


class TrxEncoderTest(torch.nn.Module):
    def forward(self, x):
        return x


def test_dropout_encoder():
    drop_encoder = DropoutEncoder(p=0.5)
    x = torch.rand(256, 100)

    drop_encoder.eval()
    assert torch.equal(x, drop_encoder(x))

    drop_encoder.train()
    assert x.shape == drop_encoder(x).shape


def test_squeeze():
    x = torch.rand(256, 100, 1)
    squeeze = Squeeze()
    y = squeeze(x)
    assert y.shape == (256, 100)


def test_mlp():
    mlp_config = {
        "hidden_layers_size": [512, 100],
        "drop_p": 0.5,
        "objective": "classification"
    }

    mlp = MLP(512, mlp_config)
    x = torch.rand(256, 512)
    y = mlp(x)
    assert y.shape == (256,)


def test_cat_layer():
    left_tail = torch.nn.Linear(100, 10)
    right_tail = torch.nn.Linear(200, 20)

    cat_layer = CatLayer(left_tail, right_tail)
    l = torch.rand(256, 100)
    r = torch.rand(256, 200)
    x = (l, r)
    y = cat_layer(x)
    assert y.shape == (256, 30)


def test_embedding_generator():
    tabular_config = {
        'num_features_count': 10,
        'cat_features_dims': [10, 10],
        'cat_emb_dim': 4
    }

    tabular_row_encoder = TabularRowEncoder(
        input_dim=tabular_config['num_features_count'] + len(tabular_config['cat_features_dims']),
        cat_dims=tabular_config['cat_features_dims'],
        cat_idxs=[x + tabular_config['num_features_count'] for x in range(len(tabular_config['cat_features_dims']))],
        cat_emb_dim=tabular_config['cat_emb_dim']
    )

    assert tabular_row_encoder.output_size == 18

    num = torch.rand(256, 10)
    cat = torch.randint(0, 10, (256, 2))
    x = torch.cat([num, cat], dim=1)
    y = tabular_row_encoder(x)

    assert y.shape == (256, 18)


def test_distribution_target_head():
    distr_target_head_config = {
        "in_size": 48,
        "num_distr_classes_pos": 4,
        "num_distr_classes_neg": 14,
        'use_gates': False,
        'pass_samples': False
    }

    distr_target_head = CombinedTargetHeadFromRnn(**distr_target_head_config)
    x = torch.rand(64, 48)
    x = (x, np.ones((64,)), np.ones((64,)))
    y = distr_target_head(x)
    assert type(y) == dict and len(y) == 4
    assert y['neg_sum'].shape == y['pos_sum'].shape == (64, 1) and \
                    y['neg_distribution'].shape == (64, 14) and y['pos_distribution'].shape == (64, 4)
