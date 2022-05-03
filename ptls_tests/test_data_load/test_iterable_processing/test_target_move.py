from ptls.data_load.iterable_processing.target_move import TargetMove
import numpy as np
import torch


def get_data():
    return [
        {
            'client_id': 1,
            'trans_date': np.array([0., 1., 2., 5., 8., 9., 11., 15.]),
            'small_group': np.array([0, 1, 2, 5, 8, 9, 11, 15]),
            'target': np.array(1)
        }
    ]


def test_target_move_filter():
    target_col = 'target'
    ifilter = TargetMove(target_col)
    data_before = get_data()
    data_after = list(ifilter(data_before))[0]
    data_before = data_before[0]
    assert isinstance(data_after, tuple)
    assert len(data_after) == 2
    target_val = data_before[target_col].item()
    assert target_val == data_after[1]
    for k, v in data_before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, data_after[0][k])
        elif isinstance(v, torch.Tensor):
            assert torch.equal(v, data_after[0][k])
        else:
            assert v == data_after[0][k]
