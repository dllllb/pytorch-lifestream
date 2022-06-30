from collections import OrderedDict

import numpy as np
import pytest
import torch

from ptls.data_load.feature_dict import FeatureDict

# FeatureDict.is_seq_feature
def test_is_seq_feature_int():
    x = 1
    assert not FeatureDict.is_seq_feature('bin', x)


def test_is_seq_feature_target_int():
    x = 1
    assert not FeatureDict.is_seq_feature('target', x)


def test_is_seq_feature_float():
    x = 1.0
    assert not FeatureDict.is_seq_feature('amount', x)


def test_is_seq_feature_str():
    x = 'user_001'
    assert not FeatureDict.is_seq_feature('user_id', x)


def test_is_seq_feature_list():
    x = [1, 2, 3, 4]
    assert not FeatureDict.is_seq_feature('mcc', x)


def test_is_seq_feature_array():
    x = np.array([1, 2, 3, 4])
    assert FeatureDict.is_seq_feature('mcc', x)


def test_is_seq_feature_event_time():
    x = None
    assert FeatureDict.is_seq_feature('event_time', x)


def test_is_seq_feature_tensor():
    x = torch.Tensor([1, 2, 3, 4])
    assert FeatureDict.is_seq_feature('mcc', x)


def test_is_seq_feature_target_array():
    x = torch.Tensor([1, 2, 3, 4])
    assert not FeatureDict.is_seq_feature('target', x)


def test_seq_indexing_seq():
    x = {'mcc': torch.Tensor([1, 2, 3, 4])}
    torch.testing.assert_close(FeatureDict.seq_indexing(x, [1, 2])['mcc'], torch.Tensor([2, 3]))

# FeatureDict.seq_indexing
def test_seq_indexing_scalar():
    x = {
        'bin': 1001,
        'target_bin': 3,
        'pp': 0.10,
        'user_id': 'u_01',
        'l_mcc': [1, 2, 3, 4],
        't_mcc': torch.IntTensor([5, 6, 7, 8, 9]),
        'event_time': torch.linspace(0, 1, 5),
        'target_dist': torch.linspace(0, 1, 11),
    }
    y = FeatureDict.seq_indexing(x, [1, 2])
    assert y['bin'] == 1001
    assert y['target_bin'] == 3
    assert y['pp'] == 0.1
    assert y['user_id'] ==  'u_01'
    assert y['l_mcc'] == [1, 2, 3, 4]
    torch.testing.assert_close(y['t_mcc'], torch.IntTensor([6, 7]))
    torch.testing.assert_close(y['event_time'], torch.FloatTensor([0.25, 0.5]))
    torch.testing.assert_close(y['target_dist'], torch.arange(0, 1.1, 0.1))


# FeatureDict.get_seq_len
def test_seq_len_et():
    x = OrderedDict([
        ('target', torch.Tensor([1, 2, 3, 4, 5])),
        ('bin', 2),
        ('mcc', torch.Tensor([1, 2, 3, 4])),
        ('event_time', torch.Tensor([1, 2, 3])),
    ])
    assert FeatureDict.get_seq_len(x) == 3


def test_seq_len_seq():
    x = OrderedDict([
        ('target', torch.Tensor([1, 2, 3, 4, 5])),
        ('bin', 2),
        ('mcc', torch.Tensor([1, 2, 3, 4])),
        ('_event_time', torch.Tensor([1, 2, 3])),
    ])
    assert FeatureDict.get_seq_len(x) == 4


def test_seq_len_nolen():
    with pytest.raises(StopIteration):
        x = OrderedDict([
            ('target', torch.Tensor([1, 2, 3, 4, 5])),
            ('bin', 2),
        ])
        FeatureDict.get_seq_len(x)
