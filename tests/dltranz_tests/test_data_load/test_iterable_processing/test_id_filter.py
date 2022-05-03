import pytest
import numpy as np

from ptls.data_load.iterable_processing.id_filter import IdFilter


def get_data(id_type):
    return [{'client_id': id_type(i)} for i in range(1, 10)]


def test_int():
    i_filter = IdFilter('client_id', [1, 5, 9])
    data = i_filter(get_data(int))
    data = [x['client_id'] for x in data]
    assert data == [1, 5, 9]


def test_np():
    i_filter = IdFilter('client_id', np.array([1, 5, 9]).astype(np.int16))
    data = i_filter(get_data(np.int16))
    data = [x['client_id'] for x in data]
    assert data == [1, 5, 9]


def test_type_mismatch_int_str():
    i_filter = IdFilter('client_id', [1, 5, 9])
    data = i_filter(get_data(str))
    with pytest.raises(TypeError):
        _ = [x['client_id'] for x in data]
