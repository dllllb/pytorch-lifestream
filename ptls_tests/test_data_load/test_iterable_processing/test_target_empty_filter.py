import numpy as np

from ptls.data_load.iterable_processing.target_empty_filter import TargetEmptyFilter


def get_data():
    return [
        {'uid': 1, 'target': 0},
        {'uid': 2, 'target': 1},
        {'uid': 3, 'target': None},
        {'uid': 4, 'target': np.NaN},
        {'uid': 5, 'target': float('NaN')},
        {'uid': 6, 'target': "class_0"},
     ]


def test_filter():
    i_filter = TargetEmptyFilter('target')
    data = i_filter(get_data())
    data = [x['uid'] for x in data]
    assert data == [1, 2, 6]
