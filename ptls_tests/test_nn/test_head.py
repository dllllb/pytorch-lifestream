"""These test are inspired by `docs/nn/head.md`
"""

import torch

from ptls.nn import Head


def test_empty_layer():
    x = torch.randn(10, 8)
    model = Head()
    y = model(x)
    torch.testing.assert_allclose(x, y)


def test_l2_norm():
    x = torch.randn(10, 8)
    model = Head(use_norm_encoder=True)
    y = model(x)
    assert y.size() == (10, 8)


def test_binary_classification():
    x = torch.randn(10, 8)
    model = Head(objective='classification', input_size=8)
    y = model(x)
    assert y.size() == (10,)


def test_multiclass_classification():
    x = torch.randn(10, 8)
    model = Head(objective='classification', input_size=8, num_classes=3)
    y = model(x)
    assert y.size() == (10, 3)


def test_mlp_binary_classification():
    x = torch.randn(10, 8)
    model = Head(objective='classification', input_size=8, hidden_layers_sizes=[12, 36])
    y = model(x)
    assert y.size() == (10,)


def test_regression():
    x = torch.randn(10, 8)
    model = Head(objective='regression', input_size=8)
    y = model(x)
    assert y.size() == (10,)
    model = Head(objective='regression', input_size=8, num_classes=3)
    y = model(x)
    assert y.size() == (10, 3)


def test_softplus():
    x = torch.randn(10, 8)
    model = Head(objective='softplus', input_size=8)
    y = model(x)
    assert y.size() == (10,)
    model = Head(objective='softplus', input_size=8, num_classes=3)
    y = model(x)
    assert y.size() == (10, 3)
