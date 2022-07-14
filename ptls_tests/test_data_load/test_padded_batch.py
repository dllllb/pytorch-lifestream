import numpy as np
import pytest
import torch

from ptls.data_load.padded_batch import PaddedBatch


def get_pb():
    return PaddedBatch(
        payload={
            'bin': torch.IntTensor([0, 1]),
            'target_bin': torch.IntTensor([2, 3]),
            'pp': torch.FloatTensor([0.1, 0.2]),
            'user_id': np.array(['a', 'b']),
            'lists': np.array([[5, 6], [7, 8]]),
            'mcc': torch.tensor([
                [1, 2, 0, 0],
                [3, 4, 5, 6],
            ]),
            'event_time': torch.tensor([
                [1, 2, 0, 0],
                [1, 2, 3, 4],
            ]),
            'target_array': torch.tensor([
                [1, 2, 2, 5],
                [1, 2, 3, 4],
            ]),
        },
        length=torch.IntTensor([2, 4])
    )


def test_padded_batch_example():
    data = PaddedBatch(
        payload=torch.tensor([
            [1, 2, 0, 0],
            [3, 4, 5, 6],
            [7, 8, 9, 0],
        ]),
        length=torch.tensor([2, 4, 3]),
    )

    # check shape
    torch.testing.assert_close(data.payload.size(), (3, 4))

    # get first transaction
    torch.testing.assert_close(data.payload[:, 0], torch.tensor([1, 3, 7]))

    # get last transaction
    torch.testing.assert_close(data.payload[torch.arange(3), data.seq_lens - 1], torch.tensor([2, 6, 9]))

    # get all transaction flatten
    torch.testing.assert_close(data.payload[data.seq_len_mask.bool()], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]))


def test_padded_batch_seq_feature_shape():
    x = get_pb()
    B, T = x.seq_feature_shape
    assert B, T == (2, 4)


def test_padded_batch_to():
    x = get_pb()
    y = x.to('cpu')
    assert len(y) == 2


def test_padded_batch_mask_tensor_trx_embedding():
    data = PaddedBatch(torch.randn(4, 5, 3), torch.tensor([2, 5, 1, 3]))
    out = data.seq_len_mask
    exp = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
    ]).long()
    torch.testing.assert_close(out, exp)


def test_padded_batch_mask_tensor_numerical():
    data = PaddedBatch(torch.randn(4, 5), torch.tensor([2, 5, 3, 1]))
    out = data.seq_len_mask
    exp = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
    ]).long()
    torch.testing.assert_close(out, exp)


def test_padded_batch_mask_dict():
    data = PaddedBatch({'col1': torch.randn(4, 5), 'col2': torch.randn(4, 5)}, torch.tensor([4, 5, 1, 3]))
    out = data.seq_len_mask
    exp = torch.tensor([
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0]
    ]).long()
    torch.testing.assert_close(out, exp)


def test_padded_batch_mask_dict_no_sequences():
    with pytest.raises(StopIteration):
        data = PaddedBatch({'target_col1': torch.randn(4, 5), 'bin': torch.IntTensor([2, 3, 1, 2])}, torch.tensor([4, 5, 1, 3]))
        _ = data.seq_len_mask

# is seq feature
def test_padded_batch_is_seq_feature():
    x = get_pb()

    for col, is_seq in [
        ('bin', False),
        ('target_bin', False),
        ('pp', False),
        ('user_id', False),
        ('lists', False),
        ('mcc', True),
        ('event_time', True),
        ('target_array', False),
    ]:
        assert is_seq == PaddedBatch.is_seq_feature(col, x.payload[col]), col


def test_padded_batch_drop_seq_features():
    x = get_pb()
    y = x.drop_seq_features()

    for col, is_seq in [
        ('bin', False),
        ('target_bin', False),
        ('pp', False),
        ('user_id', False),
        ('lists', False),
        ('mcc', True),
        ('event_time', True),
        ('target_array', False),
    ]:
        assert is_seq != (col in y)


def test_padded_batch_keep_seq_features():
    x = get_pb()
    y = x.keep_seq_features()

    for col, is_seq in [
        ('bin', False),
        ('target_bin', False),
        ('pp', False),
        ('user_id', False),
        ('lists', False),
        ('mcc', True),
        ('event_time', True),
        ('target_array', False),
    ]:
        assert is_seq == (col in y.payload)
