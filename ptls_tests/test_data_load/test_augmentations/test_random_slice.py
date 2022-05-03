import numpy as np

from ptls.data_load.augmentations.random_slice import RandomSlice


def test_usage():
    i_filter = RandomSlice(10, 20)
    data = {'mcc': np.arange(100)}
    data = i_filter(data)
    assert 10 <= len(data['mcc']) <= 20
    assert (np.diff(data['mcc']) == 1).all()


def test_len_1():
    i_filter = RandomSlice(10, 20)
    min_len, max_len = i_filter.get_min_max(100)
    assert (min_len, max_len) == (10, 20)


def test_len_2():
    i_filter = RandomSlice(10, 20)
    min_len, max_len = i_filter.get_min_max(15)
    assert (min_len, max_len) == (10, 15)


def test_len_3():
    i_filter = RandomSlice(10, 20)
    min_len, max_len = i_filter.get_min_max(5)
    assert (min_len, max_len) == (5, 5)


def test_len_3_full():
    i_filter = RandomSlice(10, 20)
    data = {'mcc': np.arange(5)}
    data = i_filter(data)
    np.testing.assert_equal(data['mcc'], np.arange(5))


def test_len_4():
    i_filter = RandomSlice(10, 20, rate_for_min=0.8)
    min_len, max_len = i_filter.get_min_max(100)
    assert (min_len, max_len) == (10, 20)


def test_len_5():
    i_filter = RandomSlice(10, 20, rate_for_min=0.8)
    min_len, max_len = i_filter.get_min_max(15)
    assert (min_len, max_len) == (10, 15)


def test_len_6():
    i_filter = RandomSlice(10, 20, rate_for_min=0.8)
    min_len, max_len = i_filter.get_min_max(5)
    assert (min_len, max_len) == (4, 5)


def test_len_7():
    i_filter = RandomSlice(10, 20, rate_for_min=0.01)
    min_len, max_len = i_filter.get_min_max(5)
    assert (min_len, max_len) == (1, 5)

