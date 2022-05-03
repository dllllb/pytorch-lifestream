from ptls.data_load.iterable_processing.iterable_shuffle import IterableShuffle
import numpy as np


def get_data_list():
    return [{'uid': i, 'mcc': np.arange(i // 2 + 10)} for i in range(10)]


def get_data_gen():
    for i in  [{'uid': i, 'mcc': np.arange(i // 2 + 10)} for i in range(10)]:
        yield i


def get_data_with_target():
    return [({'uid': i, 'mcc': np.arange(i // 2 + 10)}, i) for i in range(10)]


def test_large_buffer_list():
    i_filter = IterableShuffle(buffer_size=20)
    data = i_filter(get_data_list())
    data = [rec['uid'] for rec in data]
    print(data)
    print(np.array(data) - np.arange(len(data)))
    assert sum(data) == 45
    data = sorted(data)
    assert data == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_gen():
    i_filter = IterableShuffle(buffer_size=20)
    data = i_filter(get_data_gen())
    data = [rec['uid'] for rec in data]
    print(data)
    print(np.array(data) - np.arange(len(data)))
    assert sum(data) == 45
    data = sorted(data)
    assert data == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_small_buffer_list():
    i_filter = IterableShuffle(buffer_size=2)
    data = i_filter(get_data_list())
    data = [rec['uid'] for rec in data]
    print(data)
    print(np.array(data) - np.arange(len(data)))
    assert sum(data) == 45
    data = sorted(data)
    assert data == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_small_buffer_with_target():
    i_filter = IterableShuffle(buffer_size=2)
    data = i_filter(get_data_with_target())
    data = [rec[0]['uid'] for rec in data]
    print(data)
    print(np.array(data) - np.arange(len(data)))
    assert sum(data) == 45
    data = sorted(data)
    assert data == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
