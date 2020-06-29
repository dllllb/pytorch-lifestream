import pytest
import torch

import dltranz.data_load.lazy_dataset as lazy_dataset
from unittest.mock import MagicMock


def test_data_size_select_01():
    lazy_dataset.glob = MagicMock(return_value=[f'{i}.csv' for i in range(10)])

    data = lazy_dataset.DataFiles(None, valid_size=0.5)
    sizes = data.size_select()
    assert sizes == (5, 5)


def test_data_size_select_02():
    lazy_dataset.glob = MagicMock(return_value=[f'{i}.csv' for i in range(10)])

    data = lazy_dataset.DataFiles(None, valid_size=0.05)
    sizes = data.size_select()
    assert sizes == (9, 1)


def test_data_size_select_03():
    lazy_dataset.glob = MagicMock(return_value=[f'{i}.csv' for i in range(2)])

    data = lazy_dataset.DataFiles(None, valid_size=0.05)
    sizes = data.size_select()
    assert sizes == (1, 1)


def test_data_size_select_04():
    lazy_dataset.glob = MagicMock(return_value=[f'{i}.csv' for i in range(1)])

    with pytest.raises(AttributeError):
        data = lazy_dataset.DataFiles(None, valid_size=0.05)


def test_data_files():
    lazy_dataset.glob = MagicMock(return_value=[f'{i}.csv' for i in range(5)])

    data = lazy_dataset.DataFiles(None, valid_size=0.45, seed=42)
    assert data.valid == ['1.csv', '4.csv']


def test_lazy_dataset_01():
    ds = lazy_dataset.LazyDataset([i for i in range(5)], lambda x: [x])
    data_loader = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=2)
    data = list(iter(data_loader))
    assert (data[0] == torch.tensor([0, 1])).all()
    assert (data[1] == torch.tensor([2, 3])).all()
    assert (data[2] == torch.tensor([4])).all()


def test_lazy_dataset_02():
    """
    files: 0, 1, 2, 3, 4
    workers:
        0: 0, 1, 2, 3, 4

    data:
        0:
            0: []
            1: [0]
            2: [0, 1]
            3: [0, 1, 2]
            4: [0, 1, 2, 3]

    """
    ds = lazy_dataset.LazyDataset([i for i in range(5)], lambda x: [f'{x}-{i}' for i in range(x)])
    data_loader = torch.utils.data.DataLoader(ds, num_workers=1, batch_size=2)
    data = list(iter(data_loader))
    assert data == [['1-0', '2-0'], ['2-1', '3-0'], ['3-1', '3-2'], ['4-0', '4-1'], ['4-2', '4-3']]


def test_lazy_dataset_03():
    """
    files: 0, 1, 2, 3, 4
    workers:
        0: 0, 3
        1: 1, 4
        2: 2

    data:
        0:
            0: []
            3: [0, 1, 2]
        1:
            1: [0]
            4: [0, 1, 2, 3]
        2:
            2: [0, 1]

    """
    ds = lazy_dataset.LazyDataset([i for i in range(5)], lambda x: [f'{x}-{i}' for i in range(x)])
    data_loader = torch.utils.data.DataLoader(ds, num_workers=3, batch_size=2)
    data = list(iter(data_loader))
    assert data == [['3-0', '3-1'], ['1-0', '4-0'], ['2-0', '2-1'], ['3-2'], ['4-1', '4-2'], ['4-3']]
