import torch

from dltranz.tabnet.tab_network import TabNet


def test_tabnet():
    tabnet_config = {
        'output_dim': 256,
    }

    tabnet = TabNet(100, **tabnet_config)

    assert tabnet.output_dim == 256

    x = torch.rand(512, 100)
    y = tabnet(x)

    assert y.shape == (512, 256)
