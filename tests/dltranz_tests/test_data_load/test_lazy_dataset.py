import torch

from dltranz.data_load.lazy_dataset import LazyDataset


def test_lazy_dataset_01():
    ds = LazyDataset([i for i in range(5)], lambda x: [x])
    data_loader = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=2)
    data = list(iter(data_loader))
    assert (data[0] == torch.tensor([0, 1])).all()
    assert (data[1] == torch.tensor([2, 3])).all()
    assert (data[2] == torch.tensor([4])).all()


def get_rows_from_file(x):
    return [f'{x}-{i}' for i in range(x)]


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
    ds = LazyDataset([i for i in range(5)], get_rows_from_file)
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
    ds = LazyDataset([i for i in range(5)], get_rows_from_file)
    data_loader = torch.utils.data.DataLoader(ds, num_workers=3, batch_size=2)
    data = list(iter(data_loader))
    assert data == [['3-0', '3-1'], ['1-0', '4-0'], ['2-0', '2-1'], ['3-2'], ['4-1', '4-2'], ['4-3']]
