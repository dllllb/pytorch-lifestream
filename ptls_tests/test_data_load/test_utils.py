import numpy as np
import pytest
import torch
from collections import OrderedDict

from ptls.data_load.utils import DictTransformer, collate_feature_dict


def test_is_seq_feature_list():
    x = [1, 2, 3, 4]
    assert not DictTransformer.is_seq_feature('mcc', x)


def test_is_seq_feature_int():
    x = 1
    assert not DictTransformer.is_seq_feature('bin', x)


def test_is_seq_feature_array():
    x = np.array([1, 2, 3, 4])
    assert DictTransformer.is_seq_feature('mcc', x)


def test_is_seq_feature_tensor():
    x = torch.Tensor([1, 2, 3, 4])
    assert DictTransformer.is_seq_feature('mcc', x)


def test_is_seq_feature_target_array():
    x = torch.Tensor([1, 2, 3, 4])
    assert not DictTransformer.is_seq_feature('target', x)


def test_seq_indexing_seq():
    x = torch.Tensor([1, 2, 3, 4])
    torch.testing.assert_close(DictTransformer.seq_indexing('mcc', x, [1, 2]), torch.Tensor([2, 3]))


def test_seq_indexing_scalar():
    x = 1001
    assert DictTransformer.seq_indexing('id', x, [1, 2]) == 1001


def test_seq_indexing_target():
    x = torch.Tensor([1, 2, 3, 4])
    torch.testing.assert_close(DictTransformer.seq_indexing('target', x, [1, 2]), torch.Tensor([1, 2, 3, 4]))


def test_seq_len_et():
    x = OrderedDict([
        ('target', torch.Tensor([1, 2, 3, 4, 5])),
        ('bin', 2),
        ('mcc', torch.Tensor([1, 2, 3, 4])),
        ('event_time', torch.Tensor([1, 2, 3])),
    ])
    assert DictTransformer.get_seq_len(x) == 3


def test_seq_len_seq():
    x = OrderedDict([
        ('target', torch.Tensor([1, 2, 3, 4, 5])),
        ('bin', 2),
        ('mcc', torch.Tensor([1, 2, 3, 4])),
        ('_event_time', torch.Tensor([1, 2, 3])),
    ])
    assert DictTransformer.get_seq_len(x) == 4


def test_seq_len_nolen():
    with pytest.raises(StopIteration):
        x = OrderedDict([
            ('target', torch.Tensor([1, 2, 3, 4, 5])),
            ('bin', 2),
        ])
        DictTransformer.get_seq_len(x)


def test_collate_feature_dict():
    batch = [
        {
            'mcc': torch.tensor([1, 2, 3, 4, 5]),
            'amount': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
            'target_int': 0,
            'target_float': 0.6,
            'target_array': torch.tensor([0.1, 0.2, 0.3]),
        },
        {
            'mcc': torch.tensor([1, 2, 3]),
            'amount': torch.tensor([0.1, 0.2, 0.3]),
            'target_int': 0,
            'target_float': 0.5,
            'target_array': torch.tensor([0.1, 0.2, 0.3]),
        },
        {
            'mcc': torch.tensor([1, 2]),
            'amount': torch.tensor([0.1, 0.2]),
            'target_int': 1,
            'target_float': 0.4,
            'target_array': torch.tensor([0.1, 0.2, 0.3]),
        },
        {
            'mcc': torch.tensor([1, 2, 3, 4]),
            'amount': torch.tensor([0.1, 0.2, 0.3, 0.4]),
            'target_int': 1,
            'target_float': 0.3,
            'target_array': torch.tensor([0.1, 0.2, 0.3]),
        },
    ]
    pb = collate_feature_dict(batch)
    torch.testing.assert_close(pb.seq_lens, torch.LongTensor((5, 3, 2, 4)))
    assert pb.payload['mcc'].size() == (4, 5)
    assert pb.payload['target_array'].size() == (4, 3)


def test_collate_feature_dict_with_arrays():
    batch = [
        {
            'a_float': torch.tensor([0.1, 0.2, 0.3]),
            'mcc': torch.tensor([1, 2, 3, 4, 5]),
            'amount': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        },
        {
            'mcc': torch.tensor([1, 2, 3]),
            'amount': torch.tensor([0.1, 0.2, 0.3]),
            'a_float': torch.tensor([0.1, 0.2, 0.3]),
        },
        {
            'mcc': torch.tensor([1, 2]),
            'amount': torch.tensor([0.1, 0.2]),
            'a_float': torch.tensor([0.1, 0.2, 0.3]),
        },
        {
            'mcc': torch.tensor([1, 2, 3, 4]),
            'amount': torch.tensor([0.1, 0.2, 0.3, 0.4]),
            'a_float': torch.tensor([0.1, 0.2, 0.3]),
        },
    ]
    pb = collate_feature_dict(batch, array_cols=['a_float'])
    torch.testing.assert_close(pb.seq_lens, torch.LongTensor((5, 3, 2, 4)))
    assert pb.payload['a_float'].size() == (4, 3)


def test_collate_feature_dict_inconsistent_new():
    batch = [
        {
            'mcc': torch.tensor([1, 2, 3, 4, 5]),
            'amount': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
            'target_float': 0.6,
        },
        {
            'mcc': torch.tensor([1, 2, 3]),
            'amount': torch.tensor([0.1, 0.2, 0.3]),
            'target_int': 0,
            'target_float': 0.5,
        },
    ]
    with pytest.raises(AssertionError):
        pb = collate_feature_dict(batch)

def test_collate_feature_dict_inconsistent_missing():
    batch = [
        {
            'mcc': torch.tensor([1, 2, 3, 4, 5]),
            'amount': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
            'target_int': 0,
            'target_float': 0.5,
        },
        {
            'mcc': torch.tensor([1, 2, 3]),
            'amount': torch.tensor([0.1, 0.2, 0.3]),
            'target_int': 0,
        },
    ]
    with pytest.raises(AssertionError):
        pb = collate_feature_dict(batch)


def test_collate_feature_dict_drop_features():
    batch = [
        {
            'mcc': torch.tensor([1, 2, 3, 4, 5]),
            'amount': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
            'col_id': 'a',
        },
        {
            'mcc': torch.tensor([1, 2, 3]),
            'amount': torch.tensor([0.1, 0.2, 0.3]),
            'col_id': 'b',
        },
        {
            'mcc': torch.tensor([1, 2, 3, 4]),
            'amount': torch.tensor([0.1, 0.2, 0.3, 0.4]),
            'col_id': 'b',
        },
    ]
    pb = collate_feature_dict(batch, array_cols=['target_array'])
    torch.testing.assert_close(pb.seq_lens, torch.LongTensor((5, 3, 4)))
    assert 'mcc' in pb.payload.keys()
    assert 'amount' in pb.payload.keys()
    assert 'col_id' not in pb.payload.keys()