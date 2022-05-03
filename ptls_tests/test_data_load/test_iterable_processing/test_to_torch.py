from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch
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
    data_after = list(ToTorch()(data_before))[0]
    data_before = data_before[0]
    for k, v in data_after.items():
        casted = torch.from_numpy(data_before[k]) if isinstance(data_before[k], np.ndarray) else data_before[k]
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, casted)
        else:
            assert v == casted
