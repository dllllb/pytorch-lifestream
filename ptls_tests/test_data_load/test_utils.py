import numpy as np
import pytest
import torch

from ptls.data_load.utils import DictTransformer, collate_feature_dict


def test_is_seq_feature_list():
    x = [1, 2, 3, 4]
    assert not DictTransformer.is_seq_feature(x)


def test_is_seq_feature_array():
    x = np.array([1, 2, 3, 4])
    assert DictTransformer.is_seq_feature(x)

def test_is_seq_feature_tensor():
    x = torch.Tensor([1, 2, 3, 4])
    assert DictTransformer.is_seq_feature(x)


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


def test_collate_feature_dict_with_arrays():
    batch = [
        {
            'target_array': torch.tensor([0.1, 0.2, 0.3]),
            'mcc': torch.tensor([1, 2, 3, 4, 5]),
            'amount': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
            'target_int': 0,
            'target_float': 0.6,
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
    pb = collate_feature_dict(batch, array_cols=['target_array'])
    torch.testing.assert_close(pb.seq_lens, torch.LongTensor((5, 3, 2, 4)))
    assert 'mcc' in pb.payload.keys()
    assert 'amount' in pb.payload.keys()
    assert 'target_int' in pb.payload.keys()
    assert 'target_float' in pb.payload.keys()
    assert 'target_array' in pb.payload.keys()


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