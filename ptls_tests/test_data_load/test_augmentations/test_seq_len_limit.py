from ptls.data_load.augmentations.seq_len_limit import SeqLenLimit
import numpy as np


def test_tail_no_limit():
    i_filter = SeqLenLimit(max_seq_len=10)
    data = {'mcc': np.arange(8)}
    data = i_filter(data)
    np.testing.assert_equal(data['mcc'], np.arange(8))


def test_tail_with_limit():
    i_filter = SeqLenLimit(max_seq_len=4)
    data = {'mcc': np.arange(8)}
    data = i_filter(data)
    np.testing.assert_equal(data['mcc'], np.array([4, 5, 6, 7]))


def test_head_no_limit():
    i_filter = SeqLenLimit(max_seq_len=10, strategy='head')
    data = {'mcc': np.arange(8)}
    data = i_filter(data)
    np.testing.assert_equal(data['mcc'], np.arange(8))


def test_head_with_limit():
    i_filter = SeqLenLimit(max_seq_len=4, strategy='head')
    data = {'mcc': np.arange(8)}
    data = i_filter(data)
    np.testing.assert_equal(data['mcc'], np.array([0, 1, 2, 3]))


def test_random_no_limit():
    i_filter = SeqLenLimit(max_seq_len=10, strategy='random')
    data = {'mcc': np.arange(8)}
    data = i_filter(data)
    np.testing.assert_equal(data['mcc'], np.arange(8))


def test_random_with_limit():
    i_filter = SeqLenLimit(max_seq_len=2, strategy='random')
    data = {'mcc': np.arange(8)}
    data = i_filter(data)
    assert len(data['mcc']) == 2
    assert data['mcc'][0] + 1 == data['mcc'][1]
