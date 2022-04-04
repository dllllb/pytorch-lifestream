from dltranz.data_load.iterable_processing.to_torch_tensor import ToTorch
import numpy as np
import torch


def get_data():
    return [
        {
            'client_id': 1,
            'trans_date': np.array([0., 1., 2., 5., 8., 9., 11., 15.]),
            'small_group': np.array([0, 1, 2, 5, 8, 9, 11, 15])
        }
    ]


def test_to_torch_filter():
    data_before = get_data()
    data_after = ToTorch(data_before)
    assert isinstanse(data_after, torch.Tensor)
    casted = torch.from_numpy(data_before) if isinstance(data_before, np.ndarray) else data_before
    assert data_after == casted
