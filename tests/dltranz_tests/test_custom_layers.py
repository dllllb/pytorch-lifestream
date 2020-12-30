import torch

from dltranz.custom_layers import DropoutEncoder, Squeeze, CatLayer, MLP


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
