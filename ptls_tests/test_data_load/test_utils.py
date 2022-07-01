import numpy as np
import pytest
import torch

from ptls.data_load.utils import collate_feature_dict


def test_collate_feature_dict():
    batch = [
        {
            'bin': 0,
            'target_bin': 2,
            'pp': 0.6,
            'user_id': 'a',
            'lists': [1, 2, 3],
            'mcc': torch.tensor([1, 2, 3, 4, 5]),
            'event_time': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
            'target_array': torch.tensor([0.1, 0.2, 0.3]),
        },
        {
            'bin': 4,
            'target_bin': 3,
            'pp': 0.2,
            'user_id': 'b',
            'lists': [4, 1, 2, 3],
            'mcc': torch.tensor([1, 2]),
            'event_time': torch.tensor([0.3, 0.4]),
            'target_array': torch.tensor([0.2, 0.1, 0.3]),
        },
        {
            'bin': 5,
            'target_bin': 0,
            'pp': 0.1,
            'user_id': 'c',
            'lists': [3, 1, 0],
            'mcc': torch.tensor([1, 3, 4, 5]),
            'event_time': torch.tensor([0.2, 0.3, 0.4, 0.5]),
            'target_array': torch.tensor([0.5, 0.2, 0.3]),
        },
    ]
    pb = collate_feature_dict(batch)
    torch.testing.assert_close(pb.seq_lens, torch.LongTensor((5, 2, 4)))
    torch.testing.assert_close(pb.payload['bin'], torch.LongTensor([0, 4, 5]))
    torch.testing.assert_close(pb.payload['target_bin'], torch.LongTensor([2, 3, 0]))
    torch.testing.assert_close(pb.payload['pp'], torch.FloatTensor([0.6, 0.2, 0.1]))
    np.testing.assert_equal(pb.payload['user_id'], np.array(['a', 'b', 'c']))
    assert type(pb.payload['lists']) is np.ndarray
    assert pb.payload['lists'].dtype.kind == 'O'
    mcc_expected = torch.LongTensor([
        [1, 2, 3, 4, 5],
        [1, 2, 0, 0, 0],
        [1, 3, 4, 5, 0],
    ])
    torch.testing.assert_close(pb.payload['mcc'], mcc_expected)
    event_time_expected = torch.FloatTensor([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.0, 0.0, 0.0],
        [0.2, 0.3, 0.4, 0.5, 0.0],
    ])
    torch.testing.assert_close(pb.payload['event_time'], event_time_expected)
    target_array_expected = torch.FloatTensor([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.3],
        [0.5, 0.2, 0.3],
    ])
    torch.testing.assert_close(pb.payload['target_array'], target_array_expected)


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
        _ = collate_feature_dict(batch)
